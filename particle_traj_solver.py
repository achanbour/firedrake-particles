import numpy as np
import warnings
from dataclasses import dataclass

from firedrake import *
from pyop2.mpi import MPI
from firedrake.utils import IntType

from update_vom import VertexOnlyMeshUpdater
from particle_logger import ParticleLogger
from exceptions import ParticleCrossingLoopNotConverged
from particle_plotter import ParticlePlotterProtocol, ParticlePlotter

from vtk_output import write_particle_vtk, write_pvd

@dataclass
class ParticleTrajectorySolverParams:
    bary_tol: float
    abs_time_tol: float
    rel_time_tol: float
    max_iters: int
    log_level: str="info"
    plot: bool=False

class ParticleTrajectorySolver():
    def __init__(self, stepper, cell_crossing_solver, params: ParticleTrajectorySolverParams, plotter: ParticlePlotterProtocol = None, vtk_dir=None, vtk_base=None, monitor=None):
        # NOTE: velocity could be a function of time so we may need to pass time params
        # Currently this is achieved by running solve over one outer step [0, dt]

        self.stepper = stepper
        self.cell_crossing_solver = cell_crossing_solver
        
        self._params = params

        # NOTE: If dt changes across time steps, then consider moving this to solve()
        self.eff_time_tol = max(self._params.abs_time_tol, self._params.rel_time_tol * self.stepper.dt)

        self.particle_vom = self.stepper.X.function_space().mesh()
        self.parent_mesh = self.particle_vom._parent_mesh
        
        self.ref_cell = self.parent_mesh.coordinates.function_space().finat_element.cell

        self.particle_vom_updater = VertexOnlyMeshUpdater(self.particle_vom, self.parent_mesh)

        self.logger = ParticleLogger(level=self._params.log_level)

        # TODO: Move VTK plotting into ParticlePlotter?
        if plotter is not None:
            self.plotter = plotter
        elif params.plot:
            self.plotter = ParticlePlotter()
        else:
            self.plotter = None
        
        self._vtk_dir = vtk_dir
        self._vtk_frame = 0
        self._vtk_base = vtk_base

        self.monitor = monitor

        # Define persistent global particle IDs
        # These are locally stored (since dat.data hold one entry per owned particle) but their valued are globally unique.
        # Its values don't change under VOM rebuilds; only their locations (depending on the rank the particle is currently in) 
        # and their position in the array according to the current numbering
        FS_ids = FunctionSpace(self.particle_vom, "DG", 0)
        self.pids = Function(FS_ids, dtype=IntType, name="particle-ids")
        offset = self.particle_vom.comm.scan(self.num_particles_local) - self.num_particles_local # offset for current rank in global numbering
        self.pids.dat.data_wo[:] = np.arange(offset, offset + self.num_particles_local)

    @property
    def num_particles_local(self):
        # NOTE: VOM is constructed with overlap_type = (DistributedMeshOverlapType.NONE, 0) meaning that
        # a particle can only ever be a ghost if its parent cell is a ghost on a rank??
        return self.particle_vom.cell_set.size

    def solve(self, t_start, t_end):
        self.outer_loop_iter = 0
        
        if self.plotter:
            self.plotter.setup(self.particle_vom, self.parent_mesh)

        while t_start < t_end - 1e-12:
            self.outer_loop_iter += 1
            dt_step = min(self.stepper.dt, t_end - t_start)
            self.logger.outer_loop(self.outer_loop_iter, t_start, dt_step, t_end, self.num_particles_local)
            
            # Run the inner loop to deal with crossings
            boundary_particles_current, migrating_particles_current = self._run_inner_loop(dt_step)

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
                boundary_particles_current + migrating_particles_current,
                self.particle_vom.reference_coordinates.dat.data_ro,
                new_phys_coords.dat.data_ro
            )

            # Handle boundary particles and update the particle VOM
            n_rebuild_global = self.particle_vom.comm.allreduce(len(boundary_particles_current) + len(migrating_particles_current), op=MPI.SUM)
            if n_rebuild_global > 0:
                # XXX: all ranks call; ranks with no absorbed particles pass boundary_particles_current=[]
                self.particle_vom_updater.rebuild_vom(absorbed_vom_indices=boundary_particles_current, 
                                                      new_coords=new_phys_coords)

                # Invalidate the stepper's cached parloops    
                self.stepper.invalidate()
            else:
                # Update the VOM's coordinates (topology stays the same)
                self.particle_vom.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro

            if self._vtk_dir is not None:
                write_particle_vtk(self._vtk_base.format(self._vtk_frame), self.particle_vom, self.stepper.v)
                self._vtk_frame += 1

            t_start += dt_step
            

            if self.plotter:
                self.plotter.update(self.particle_vom)

            if self.monitor is not None:
                self.monitor(t_start, self)
        
        if self.plotter:
            self.plotter.close()
        
        if self._vtk_dir is not None:
            write_pvd("particles.pvd",
                      self._vtk_base,
                      self._vtk_frame)
    
            
        return t_start, self.pids.dat.data_ro.copy()

        
    def _run_inner_loop(self, dt_step):
        self.inner_loop_iter = 0
        self.particles_inner_loop_iter = np.zeros(self.num_particles_local, dtype=int)

        dt_remaining = np.full(self.num_particles_local, dt_step)
        new_ref_pos = self.particle_vom.reference_coordinates.dat.data_ro.copy()

        boundary_particles_current = []
        migrating_particles_current = []

        while self.inner_loop_iter < self._params.max_iters:
            particles_have_dt_remaining = dt_remaining > self.eff_time_tol

            any_remaining = self.particle_vom.comm.allreduce(any(particles_have_dt_remaining), op=MPI.LOR)
            if not any_remaining:
                break

            self.inner_loop_iter += 1

            particles_with_dt_remaining_idxs = np.where(particles_have_dt_remaining)[0]
            self.particles_inner_loop_iter[particles_with_dt_remaining_idxs] += 1

            self.stepper.dt_fn.dat.zero()
            self.stepper.dt_fn.dat.data_wo[particles_with_dt_remaining_idxs] = dt_remaining[particles_with_dt_remaining_idxs]

            # -- Capture initial state
            self.stepper.begin_substep()

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

            # record which dt was used to advance the particles
            dt_used = np.zeros(self.num_particles_local, dtype=float)

            # Process passed particles
            if len(particles_passed_global_idxs) > 0:
                dt_used[particles_passed_global_idxs] = dt_remaining[particles_passed_global_idxs]
                dt_remaining[particles_passed_global_idxs] = 0
                new_ref_pos[particles_passed_global_idxs] = candidate_ref_pos.dat.data_ro[particles_passed_global_idxs]

                # Register coordinates
                new_ref_pos[particles_passed_global_idxs] = candidate_ref_pos.dat.data_ro[particles_passed_global_idxs]

                self.logger.print_particles("Passed particles",
                                            {
                                                "X_new": new_ref_pos[particles_passed_global_idxs]
                                            },
                                            indices=particles_passed_global_idxs, 
                                            level="info")
            
            parent_cells = self.particle_vom.topology.cell_parent_cell_list.copy()
            new_parent_cells = parent_cells.copy()
        
            # Process failed particles
            # NOTE: all ranks enter bisection whenever any rank has failing particles (bisection calls stepper.solve() which calls assemble() which is collective)
            n_failed_global = self.particle_vom.comm.allreduce(len(particles_failed_global_idxs), op=MPI.SUM)
            if n_failed_global > 0:
                # Use the cell crossing solver to solve for crossing time and crossing position of each failed particle
                t_cross, X_cross, bary_cross = self.cell_crossing_solver.solve(
                    self.stepper,
                    self.ref_cell,
                    particles_failed_global_idxs,
                    dt_remaining[particles_failed_global_idxs],
                    bary_tol=self._params.bary_tol,
                    time_tol=self.eff_time_tol,
                )
                dt_used[particles_failed_global_idxs] = t_cross

                # Decrement dt_remaining by t_cross
                dt_remaining[particles_failed_global_idxs] -= t_cross

                self.logger.print_particles("Failed particles",
                                            {
                                                "dt_remaining": dt_remaining[particles_failed_global_idxs],
                                                "t_cross": t_cross,
                                                "bary_coords_cross": bary_cross,
                                                "X_cross": X_cross,
                                            }, 
                                            indices=particles_failed_global_idxs,
                                            level="info")

                # Determine which edge each particle crossed
                crossed_edge_idxs = np.full(len(particles_with_dt_remaining_idxs), None, dtype=object)

                # Check that the selected edge agrees with the direction of the particle's motion (given by the velocity field)
                for i, pid in enumerate(particles_failed_local_idxs):
                    crossed_edge = int(np.argmin(abs(bary_cross[i])))
                    crossed_edge_normal = self.ref_cell.compute_reference_normal(1, crossed_edge)
                    
                    # XXX: stepper._v_ref has already been evaluated in stepper.step()
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
                    neighbours  = self.parent_mesh.topology.cell_facet_neighbours.data[parent_cell]
                    crossed_edge = crossed_edge_idxs[particles_failed_local_idxs[i]]

                    # XXX: may be -1 or owned/ghost cell otherwise.
                    next_cell = self.parent_mesh.topology.cell_facet_neighbours.data[parent_cell, crossed_edge]

                    assert next_cell == -1 or next_cell in neighbours, \
                        f"particle {pid}: non-adjacent {next_cell} from {parent_cell} (nbrs {neighbours})"

                    if next_cell == -1:
                        # exterior boundary -> absorb particle
                        boundary_particles_current.append(pid)
                        dt_remaining[pid] = 0
                        warnings.warn(f"Particle {pid} crossed an exterior boundary from cell {parent_cell}")
                    elif next_cell > self.parent_mesh.cell_set.size:
                        # ghost cell -> migrate particle to rank that owns the new parent cell
                        migrating_particles_current.append(pid)
                        new_parent_cells[pid, 0] = next_cell
                    else:
                        # owned cell (same rank) -> update coords. + continue moving
                        new_parent_cells[pid, 0] = next_cell

                    A_facet, b_facet = self.parent_mesh.topology.cell_facet_coord_transforms
                    new_ref_pos[pid] = A_facet.data[parent_cell, crossed_edge] @ X_cross[i] + b_facet.data[parent_cell, crossed_edge]
                
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
            
            # Change dt to match the step consumed by each particle
            self.stepper.dt_fn.dat.zero()
            self.stepper.dt_fn.dat.data_wo[particles_with_dt_remaining_idxs] = dt_used[particles_with_dt_remaining_idxs]
            self.stepper.finalize_substep()
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
        return boundary_particles_current, migrating_particles_current