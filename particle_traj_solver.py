import numpy as np
from firedrake import *
import warnings
from dataclasses import dataclass

from update_vom import VertexOnlyMeshUpdater
from particle_logger import ParticleLogger
from exceptions import ParticleCrossingLoopNotConverged
from particle_plotter import ParticlePlotterProtocol, ParticlePlotter

@dataclass
class ParticleTrajectorySolverParams:
    bary_tol: float
    abs_time_tol: float
    rel_time_tol: float
    max_iters: int
    log_level: str="info"
    plot: bool=False

class ParticleTrajectorySolver():
    def __init__(self, stepper, cell_crossing_solver, params: ParticleTrajectorySolverParams, plotter: ParticlePlotterProtocol = None):
        # velocity could be a function of time so may need to pass time params

        self.stepper = stepper
        self.cell_crossing_solver = cell_crossing_solver
        
        self._params = params

        # If dt changes across time steps, then consider moving this to solve.
        self.eff_time_tol = max(self._params.abs_time_tol, self._params.rel_time_tol * self.stepper.dt)

        self.particle_vom = self.stepper.X.function_space().mesh()
        self.parent_mesh = self.particle_vom._parent_mesh
        
        self.ref_cell = self.parent_mesh.coordinates.function_space().finat_element.cell

        self.particle_vom_updater = VertexOnlyMeshUpdater(self.particle_vom, self.parent_mesh)

        self.logger = ParticleLogger(level=self._params.log_level)

        if plotter is not None:
            self.plotter = plotter
        elif params.plot:
            self.plotter = ParticlePlotter()
        else:
            self.plotter = None

    @property
    def num_particles(self):
        return self.particle_vom.num_vertices()

    def solve(self, t_start, t_end):
        particle_ids = np.arange(self.particle_vom.num_vertices())
        self.outer_loop_iter = 0
        
        if self.plotter:
            self.plotter.setup(self.particle_vom, self.parent_mesh)

        while t_start < t_end - 1e-12:
            self.outer_loop_iter += 1
            self.logger.outer_loop(self.outer_loop_iter, t_start, self.stepper.dt, t_end, self.num_particles)

            # Run the inner time step loop
            boundary_particles_current = self._run_inner_loop()

            # Recompute physical coordinates
            new_phys_coords = assemble(
                interpolate(
                    SpatialCoordinate(self.parent_mesh), self.particle_vom.coordinates.function_space()
                )
            )

            self.logger.outer_summary(
                self.outer_loop_iter,
                self.inner_loop_iter,
                self.particles_inner_loop_iter,
                boundary_particles_current,
                self.particle_vom.reference_coordinates.dat.data_ro,
                new_phys_coords.dat.data_ro
            )

            # Handle boundary particles and update the particle VOM
            if len(boundary_particles_current) > 0:
                # Rebuild the VOM topologically
                old_particle_ids = particle_ids
                reorder_map = self.particle_vom_updater.rebuild_vom(absorbed_vom_indices=boundary_particles_current, new_coords=new_phys_coords)

                # Invalidate parloops as Function Spaces have been redefined
                # TODO: if we preserve the VOM topology, then we can preserve Functions and Function Spaces defined on it.
                # For now, rebuild all fields stored in the stepper.
                self.stepper.invalidate()

                # Update particle_ids so that it is always in sync with the latest particle set
                particle_ids = old_particle_ids[reorder_map]
            else:
                # Update the VOM's coordinates (topology stays the same)
                self.particle_vom.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro

            t_start += self.stepper.dt

            if self.plotter:
                self.plotter.update(self.particle_vom)
        
        if self.plotter:
            self.plotter.close()
            
        return t_start, particle_ids

        
    def _run_inner_loop(self):
        self.inner_loop_iter = 0
        self.particles_inner_loop_iter = np.zeros(self.num_particles, dtype=int)

        dt_remaining = np.full(self.num_particles, self.stepper.dt)
        new_ref_pos = self.particle_vom.reference_coordinates.dat.data_ro.copy()
        boundary_particles_current = []

        while self.inner_loop_iter < self._params.max_iters:
            particles_have_dt_remaining = dt_remaining > self.eff_time_tol
            if not any(particles_have_dt_remaining):
                break

            self.inner_loop_iter += 1

            particles_with_dt_remaining_idxs = np.where(particles_have_dt_remaining)[0]
            self.particles_inner_loop_iter[particles_with_dt_remaining_idxs] += 1

            self.stepper.dt_fn.dat.zero()
            self.stepper.dt_fn.dat.data_wo[particles_with_dt_remaining_idxs] = dt_remaining[particles_with_dt_remaining_idxs]

            # Compute candidate reference positions by executing a full step
            candidate_ref_pos = self.stepper.step()
            candidate_bary_coords = self.ref_cell.compute_barycentric_coordinates(candidate_ref_pos.dat.data_ro)
            
            # Split particles into two subsets (crossed/not_crossed) depending on whether they have left their containing cell
            # Local set indexes into the set particles_with_remaining_dt
            # Global set indexes into the full set of particles
            passed_mask = np.all(candidate_bary_coords[particles_with_dt_remaining_idxs] >= -self._params.bary_tol, axis=1)
            
            particles_passed_local_idxs = np.where(passed_mask)[0]
            particles_failed_local_idxs = np.where(~passed_mask)[0]
            
            particles_passed_global_idxs = particles_with_dt_remaining_idxs[particles_passed_local_idxs]
            particles_failed_global_idxs = particles_with_dt_remaining_idxs[particles_failed_local_idxs]
            
            self.logger.inner_loop(self.inner_loop_iter, particles_with_dt_remaining_idxs, particles_passed_global_idxs, particles_failed_global_idxs)

            # Process passed particles
            if len(particles_passed_global_idxs) > 0:
                # Set dt_remaining to 0
                dt_remaining[particles_passed_global_idxs] = 0

                # Register coordinates
                new_ref_pos[particles_passed_global_idxs] = candidate_ref_pos.dat.data_ro[particles_passed_global_idxs]

                self.logger.print_particles("Passed particles",
                                            {
                                                "X_new": new_ref_pos[particles_passed_global_idxs]
                                            },
                                            indices=particles_passed_global_idxs, level="info")
            
            parent_cells = self.particle_vom.topology.cell_parent_cell_list.copy()
            new_parent_cells = parent_cells.copy()
        
            # Process failed particles
            if len(particles_failed_global_idxs) > 0:
                # Use the cell crossing solver to solve for crossing time and crossing position of each failed particle
                t_cross, X_cross, bary_cross = self.cell_crossing_solver.solve(
                    self.stepper,
                    self.ref_cell,
                    particles_failed_global_idxs,
                    dt_remaining[particles_failed_global_idxs],
                    bary_tol=self._params.bary_tol,
                    time_tol=self.eff_time_tol,
                )

                # Decrement dt_remaining by t_cross
                dt_remaining[particles_failed_global_idxs] -= t_cross

                self.logger.print_particles("Failed particles",
                                            {
                                                "dt_remaining": dt_remaining[particles_failed_global_idxs],
                                                "t_cross": t_cross,
                                                "bary_coords_cross": bary_cross,
                                                "X_cross": X_cross 
                                            }, 
                                            indices=particles_failed_global_idxs)

                # Determine which edge each particle crossed
                crossed_edge_idxs = np.full(len(particles_with_dt_remaining_idxs), None, dtype=object)

                # Check that the selected edge agrees with the direction of the particle's motion (given by the velocity field)
                for i, pid in enumerate(particles_failed_local_idxs):
                    crossed_edge = int(np.argmin(abs(bary_cross[i])))
                    crossed_edge_normal = self.ref_cell.compute_reference_normal(1, crossed_edge)
                    
                    # NOTE: stepper._v_ref has already been evaluated in stepper.step()
                    # so either re-assemble as done here or stash and reuse the results from the previous evaluation
                    v_ref = self.stepper.v_ref.dat.data_ro[particles_failed_global_idxs[i]]

                    # TODO: Test this on edge cases
                    if np.dot(crossed_edge_normal, v_ref) <= 0:
                        for other_edge in range(len(bary_cross[i])):
                            if other_edge == crossed_edge:
                                continue
                            other_edge_normal = self.ref_cell.compute_reference_normal(1, other_edge)
                            if np.dot(other_edge_normal, v_ref) > 0:
                                crossed_edge = other_edge
                                break
                    crossed_edge_idxs[pid] = crossed_edge

                # Identify the neighbouring cell for each particle and its reference coordinates in that new cell
                for i, pid in enumerate(particles_failed_global_idxs):
                    parent_cell = parent_cells[pid, 0]

                    crossed_edge = crossed_edge_idxs[particles_failed_local_idxs[i]]
                    next_cell = self.parent_mesh.topology.cell_facet_neighbours.data[parent_cell, crossed_edge]

                    if next_cell is None or next_cell == -1:
                        # Particle hits an exterior boundary
                        new_parent_cells[pid, 0] = parent_cell
                        boundary_particles_current.append(pid)
                        dt_remaining[pid] = 0
                        warnings.warn(f"Particle {pid} attempted to cross an exterior boundary facet from cell {parent_cell}")
                    else:
                        new_parent_cells[pid, 0] = next_cell
                    
                    A_facet_coord_transform, b_facet_coord_transform = self.parent_mesh.topology.cell_facet_coord_transforms
                    new_ref_pos[pid] = A_facet_coord_transform.data[parent_cell, crossed_edge] @ X_cross[i] + b_facet_coord_transform.data[parent_cell, crossed_edge]
                
                self.logger.print_particles("Failed particles - Cell transitions",
                                            {
                                                "parent_cell": parent_cells[particles_failed_global_idxs, 0],
                                                "crossed_edge": crossed_edge_idxs[particles_failed_local_idxs],
                                                "new_parent_cell": new_parent_cells[particles_failed_global_idxs, 0],
                                                "X_new": new_ref_pos[particles_failed_global_idxs]
                                            },
                                            indices=particles_failed_global_idxs,
                                            level="info")

            # Update the particle VOM
            self.particle_vom_updater.update_ref_view(new_parent_cells, new_ref_pos)
        else:
            breakpoint()
            particles_not_resolved_idxs = np.where(dt_remaining > self.eff_time_tol)[0]
            self.logger.print_particles("Non-resolved particles",
                                        {
                                            "dt_remaining": dt_remaining[particles_not_resolved_idxs]
                                        },
                                        indices=particles_not_resolved_idxs,
                                        level="info")
            raise ParticleCrossingLoopNotConverged(
                    f"Cell crossings could not be resolved within {self._params.max_iters} iterations."
                )
        return boundary_particles_current