import numpy as np
from firedrake import *
import warnings

from dataclasses import dataclass
from update_vom import VertexOnlyMeshUpdater

from exceptions import ParticleCrossingLoopNotConverged

@dataclass
class ParticleTrajectorySolverParams:
    bary_tol: float
    abs_time_tol: float
    rel_time_tol: float
    max_iters: int

class ParticleTrajectorySolver():
    def __init__(self, stepper, cell_crossing_solver, params: ParticleTrajectorySolverParams):
        # velocity could be a function of time so may need to pass time params

        self.stepper = stepper
        self.cell_crossing_solver = cell_crossing_solver
        
        self._params = params
        self.eff_time_tol = max(self._params.abs_time_tol, self._params.rel_time_tol * dt)

        self.particle_vom = self.stepper.X.mesh()
        self.num_particles = self.particle_vom.num_vertices()
        self.parent_mesh = self.particle_vom._parent_mesh
        
        self.ref_cell = self.parent_mesh.coordinates.function_space().finat_element.cell

        self.particle_vom_updater = VertexOnlyMeshUpdater(self.particle_vom, self.parent_mesh)
        

    def solve(self, t_start, t_end):
        particle_ids = np.arange(self.num_particles)
        boundary_particles = []

        while t_start < t_end - 1e-12:
            # Run the inner time step loop
            # TODO: extract initial dt from stepper (not time_remaining)
            boundary_particles_current = self._run_inner_loop(dt)

            # Recompute physical coordinates
            new_phys_coords = assemble(
                interpolate(
                    SpatialCoordinate(self.parent_mesh), self.particle_vom.coordinates.function_space()
                )
            )

            # Handle boundary particles and update the particle VOM
            if len(boundary_particles_current) > 0:
                # Need to rebuild the VOM topologically
                for pid in boundary_particles_current:
                    boundary_particles.append(particle_ids[pid])
                
                self.particle_vom_updater.rebuild_vom(absorbed_vom_indices=boundary_particles_current, new_coords=new_phys_coords)

                self.stepper.invalidate()
                # TODO: rebuild all fields

                # Update particle_ids so that it is always in sync with the latest particle set
                survived = np.ones(self.num_particles, dtype=bool)
                survived[boundary_particles_current] = False
                particle_ids = particle_ids[survived]

            else:
                # Update the VOM's coordinates (topology stays the same)
                self.particle_vom.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro

            t_start += dt
            
            return t_start, boundary_particles

        
    def _run_inner_loop(self):
        dt_remaining = np.full(self.num_particles, dt)
        new_ref_pos = self.particle_vom.reference_coordinates.data_ro.copy()

        inner_loop_iter = 0
        particles_with_dt_remaining_iters = np.zeros(self.num_particles, dtype=int)

        boundary_particles_current = []

        while inner_loop_iter < self._params.max_iters:
            particles_with_dt_remaining = particles_with_dt_remaining > self.eff_time_tol
            if not any(particles_with_dt_remaining):
                break

            inner_loop_iter += 1
            particles_with_dt_remaining_idxs = np.where(particles_with_dt_remaining)[0]
            particles_with_dt_remaining_iters[particles_with_dt_remaining_idxs] += 1

            self.stepper.dt.dat.zero()
            self.stepper.dt.data_wo[particles_with_dt_remaining_idxs] = dt_remaining[particles_with_dt_remaining_idxs]
            
            # TODO: Re-evaluate all fields on the current particle VOM
            self.stepper.refresh_fields()

            # Compute candidate reference positions by executing a full step
            candidate_ref_pos = self.stepper.step()
            candidate_bary_coords = self.ref_cell.compute_barycentric_coordinates(candidate_ref_pos)
            
            # Split particles into two subsets (crossed/not_crossed) depending on whether they have left their containing cell
            # Local set indexes into the set particles_with_remaining_dt
            # Global set indexes into the full set of particles
            passed_mask = np.all(candidate_bary_coords[particles_with_dt_remaining_idxs] >= self._params.bary_tol, axis=1)
            
            particles_passed_local_idxs = np.where(passed_mask)[0]
            particles_failed_local_idxs = np.where(~passed_mask)[0]
            
            particles_passed_global_idxs = particles_with_dt_remaining_idxs[particles_passed_local_idxs]
            particles_failed_global_idxs = particles_with_dt_remaining_idxs[particles_failed_local_idxs]
            
            # Process passed particles
            if len(particles_passed_global_idxs) > 0:
                # Set dt_remaining to 0
                dt_remaining[particles_passed_global_idxs] = 0

                # Register coordinates
                new_ref_pos[particles_passed_global_idxs] = candidate_ref_pos.dat.data_ro[particles_passed_global_idxs]
            
            parent_cells = self.particle_vom.cell_parent_cell_list.copy()
            new_parent_cells = parent_cells.copy()
        
            # Process failed particles
            if len(particles_failed_global_idxs) > 0:
                # Use the cell crossing solver to solve for crossing time and crossing position of each failed particle
                t_cross, X_cross, bary_cross = self.cell_crossing_solver(
                    self.stepper,
                    self.ref_cell,
                    particles_failed_global_idxs,
                    bary_tol=self._params.bary_tol,
                    time_tol=self.eff_time_tol,
                    # max_iters = max_iters -- set by user when instantiating the solver?
                )

                # Determine which edge each particle crossed
                crossed_edge_idxs = np.full(len(particles_with_dt_remaining_idxs), None, dtype=object)

                # Check that the selected edge agrees with the direction of the particle's motion (given by the velocity field)
                for i, pid in enumerate(particles_failed_local_idxs):
                    crossed_edge = int(np.argmin(abs(bary_cross[i])))
                    crossed_edge_normal = self.ref_cell.compute_reference_normal(1, crossed_edge)
                    
                    # TODO: Is velocity always defined in the stepper? Is invJ also owned by the stepper?
                    v_ref = invJ_vom.dat.data_ro[particles_failed_global_idxs[i]] @ self.stepper.v.dat.data_ro[particles_failed_global_idxs[i]]

                    if np.dot(crossed_edge_normal, v_ref) <= 0:
                        for other_edge in range(self.ref_cell.get_topology()[1]):
                            if other_edge == crossed_edge:
                                pass
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
                        # Particle hit an exterior boundary
                        new_parent_cells[pid, 0] = parent_cell
                        boundary_particles_current.append(pid)
                        dt_remaining[pid] = 0
                        warnings.warn(f"Particle {pid} attempted to cross an exterior boundary facet from cell {parent_cell}")
                    else:
                        new_parent_cells[pid, 0] = next_cell
                    
                    A_crossed_edge, b_crossed_edge = self.parent_mesh.topology.cell_facet_coord_transforms
                    new_ref_pos[pid] = A_crossed_edge[parent_cell, crossed_edge] @ X_cross[i] + b_crossed_edge[parent_cell, crossed_edge]
                
                # Update the particle VOM
                self.particle_vom_updater.update_ref_view(new_parent_cells, new_ref_pos)
        else:
            raise ParticleCrossingLoopNotConverged(
                    f"Cell crossings could not be resolved within {self._params.max_iters} iterations."
                )
        return boundary_particles_current



