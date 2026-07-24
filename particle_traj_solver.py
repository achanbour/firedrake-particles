import numpy as np
import warnings
from dataclasses import dataclass

from firedrake import *
from pyop2.mpi import MPI
from firedrake.utils import IntType

from mutate_vom import VertexOnlyMeshUpdater
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
    def __init__(self, stepper, cell_crossing_solver, params: ParticleTrajectorySolverParams, plotter: ParticlePlotterProtocol = None, monitor=None):
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

        self.particle_vom_updater = VertexOnlyMeshUpdater(self.particle_vom)

        self.logger = ParticleLogger(level=self._params.log_level)

        if plotter is not None:
            self.plotter = plotter
        elif params.plot:
            self.plotter = ParticlePlotter()
        else:
            self.plotter = None

        self.monitor = monitor

        DG0 = FunctionSpace(self.particle_vom, "DG", 0)

        # Define a per-particle remainining time-step Function
        self.dt_remaining = Function(DG0, name="dt-remaining")

        self._fields = (self.dt_remaining, *self.stepper.fields)

    @property
    def num_particles_local(self):
        # NOTE: VOM is constructed with overlap_type = (DistributedMeshOverlapType.NONE, 0) meaning that
        # a particle can only ever be a ghost if its parent cell is a ghost on a rank??
        return self.particle_vom.cell_set.size

    def solve(self, t_start, t_end):
        self.outer_loop_iter = 0
        self.absorbed_pids = []
        
        if self.plotter:
            self.plotter.setup(self.particle_vom, self.parent_mesh)

        while t_start < t_end - 1e-12:
            self.outer_loop_iter += 1
            dt_step = min(self.stepper.dt, t_end - t_start)
            self.logger.outer_loop(self.outer_loop_iter, t_start, dt_step, t_end, self.num_particles_local)

            self.stepper.begin_outer_step()
            
            # Run the inner loop to deal with crossings
            self._run_inner_loop(dt_step)

            self.stepper.finalize_outer_step()

            # Recompute physical coordinates
            new_phys_coords = assemble(
                interpolate(
                    SpatialCoordinate(self.parent_mesh), self.particle_vom.coordinates.function_space()
                )
            )

            self.logger.outer_summary(
                self.outer_loop_iter,
                self.inner_loop_iter,
                self.absorbed_pids,
                self.particle_vom.reference_coordinates.dat.data_ro,
                new_phys_coords.dat.data_ro
            )


            # Update the VOM's physical coordinates
            self.particle_vom.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro

            t_start += dt_step
            
            if self.plotter:
                self.plotter.update(self.particle_vom)

            if self.monitor is not None:
                self.monitor(t_start, self)

        if self.plotter:
            self.plotter.close()

        return t_start

        
    def _run_inner_loop(self, dt_step):
        self.inner_loop_iter = 0

        topo = self.particle_vom.topology

        boundary_particles_current = []
        migrating_particles_current = []

        new_ref_pos = self.particle_vom.reference_coordinates.dat.data_ro.copy()

        self.dt_remaining.dat.data_wo[:] = dt_step

        while self.inner_loop_iter < self._params.max_iters:
            dt_remaining_arr = self.dt_remaining.dat.data_ro.copy()
            dt_consumed_arr = np.zeros_like(dt_remaining_arr)

            particles_have_dt_remaining = dt_remaining_arr > self.eff_time_tol
            local_active = bool(np.any(particles_have_dt_remaining))

            if not self.particle_vom.comm.allreduce(local_active, op=MPI.LOR):
                break

            self.inner_loop_iter += 1

            active_particles_idxs = np.where(particles_have_dt_remaining)[0]

            self.stepper.dt_fn.dat.zero()
            self.stepper.dt_fn.dat.data_wo[active_particles_idxs] = dt_remaining_arr[active_particles_idxs]

            self.stepper.begin_substep()

            # Compute candidate reference positions by executing a full step
            candidate_ref_pos = self.stepper.step()
            candidate_bary_coords = self.ref_cell.compute_barycentric_coordinates(candidate_ref_pos.dat.data_ro)
            
            # Split particles into two subsets crossed/not_crossed depending on whether they have left their containing cell
            # - Passed/Failed index set corresponds the rank-local set of particles based on the current VOM ordering
            # - Active index set corresponds to the active subset given by particles_with_remaining_dt
            passed_mask = np.all(candidate_bary_coords[active_particles_idxs] >= -self._params.bary_tol, axis=1)
            
            particles_passed_active_idxs = np.where(passed_mask)[0]
            particles_failed_active_idxs = np.where(~passed_mask)[0]
            
            particles_passed_idxs = active_particles_idxs[particles_passed_active_idxs]
            particles_failed_idxs = active_particles_idxs[particles_failed_active_idxs]
            
            self.logger.inner_loop(self.inner_loop_iter, active_particles_idxs, particles_passed_idxs, particles_failed_idxs)

            # Process passed particles
            if len(particles_passed_idxs) > 0:
                dt_consumed_arr[particles_passed_idxs] = dt_remaining_arr[particles_passed_idxs]
                dt_remaining_arr[particles_passed_idxs] = 0

                # Register coordinates
                new_ref_pos[particles_passed_idxs] = candidate_ref_pos.dat.data_ro[particles_passed_idxs]

                self.logger.print_particles("Passed particles",
                                            {
                                                "X_new": new_ref_pos[particles_passed_idxs]
                                            },
                                            indices=particles_passed_idxs, 
                                            level="info")
            
            parent_cells = self.particle_vom.topology.cell_parent_cell_list.copy()[:self.num_particles_local].copy()
            new_parent_cells = parent_cells.copy()
        
            # Process failed particles
            # NOTE: all ranks enter bisection whenever any rank has failing particles (bisection calls stepper.solve() which calls assemble() which is collective)
            n_failed_global = self.particle_vom.comm.allreduce(len(particles_failed_idxs), op=MPI.SUM)
            if n_failed_global > 0:
                # Use the cell crossing solver to solve for crossing time and crossing position of each failed particle
                t_cross, X_cross, bary_cross = self.cell_crossing_solver.solve(
                    self.stepper,
                    self.ref_cell,
                    particles_failed_idxs,
                    dt_remaining_arr[particles_failed_idxs],
                    bary_tol=self._params.bary_tol,
                    time_tol=self.eff_time_tol,
                )

                dt_consumed_arr[particles_failed_idxs] = t_cross 

                # Decrement dt_remaining by t_cross
                dt_remaining_arr[particles_failed_idxs] -= t_cross

                self.logger.print_particles("Failed particles",
                                            {
                                                "dt_remaining": dt_remaining_arr[particles_failed_idxs],
                                                "t_cross": t_cross,
                                                "bary_coords_cross": bary_cross,
                                                "X_cross": X_cross,
                                            }, 
                                            indices=particles_failed_idxs,
                                            level="info")

                # Determine which edge each particle crossed
                crossed_edge_idxs = np.full(len(active_particles_idxs), None, dtype=object)

                v_ref_arr = self.stepper.v_ref.dat.data_ro

                # Check that the selected edge agrees with the direction of the particle's motion (given by the velocity field)
                for i, pid_local in enumerate(particles_failed_active_idxs):
                    crossed_edge = int(np.argmin(abs(bary_cross[i])))
                    crossed_edge_normal = self.ref_cell.compute_reference_normal(1, crossed_edge)

                    v_ref = v_ref_arr[particles_failed_idxs[i]]
                    
                    # TODO: Test this on edge cases
                    if np.dot(crossed_edge_normal, v_ref) <= 0:
                        for other_edge in range(len(bary_cross[i])):
                            if other_edge == crossed_edge:
                                continue
                            other_edge_normal = self.ref_cell.compute_reference_normal(1, other_edge)
                            if np.dot(other_edge_normal, v_ref) > 0:
                                crossed_edge = other_edge
                                break
                    crossed_edge_idxs[pid_local] = crossed_edge

                ext_mask = self.parent_mesh.topology.cell_facet_exterior_mask
                state = np.full(len(particles_failed_idxs), "", dtype=object)

                # Identify the neighbouring cell for each particle and compute its reference coordinates in that new cell
                for i, pid_local in enumerate(particles_failed_idxs):
                    parent_cell = parent_cells[pid_local, 0]
                    crossed_edge = crossed_edge_idxs[particles_failed_active_idxs[i]]
                    next_cell = self.parent_mesh.topology.cell_facet_neighbours.data_ro_with_halos[parent_cell, crossed_edge]

                    if next_cell != -1:
                        # Owned or ghost -> transfer and keep stepping
                        new_parent_cells[pid_local, 0] = next_cell
                        A_facet, b_facet = self.parent_mesh.topology.cell_facet_coord_transforms
                        new_ref_pos[pid_local] = A_facet.data_ro_with_halos[parent_cell, crossed_edge] @ X_cross[i] + b_facet.data_ro_with_halos[parent_cell, crossed_edge]
                        if next_cell < self.particle_vom.cell_set.size:
                            state[i] = "crossed into owned cell"
                        else: 
                            state[i] = "crossed into ghost cell"
                    elif ext_mask[parent_cell, crossed_edge]:
                        # At domain boundary -> absorb the particle
                        boundary_particles_current.append(pid_local)
                        dt_remaining_arr[pid_local] = 0 
                        new_ref_pos[pid_local] = X_cross[i]
                        state[i] = "absorbed"
                    else:
                        # At partition boundary (neighbour not locally known) -> hand over to the new owning rank
                        migrating_particles_current.append(pid_local)
                        new_ref_pos[pid_local] = X_cross[i]
                        state[i] = "migrated to new owning rank"
                
                self.logger.print_particles("Failed particles - Cell transitions",
                                            {
                                                "parent_cell": parent_cells[particles_failed_idxs, 0],
                                                "crossed_edge": crossed_edge_idxs[particles_failed_active_idxs],
                                                "new_parent_cell": new_parent_cells[particles_failed_idxs, 0],
                                                "state": state,
                                                "X_new": new_ref_pos[particles_failed_idxs]
                                            },
                                            indices=particles_failed_idxs,
                                            level="info")

            # Update the particle VOM
            self.particle_vom_updater.commit_reference_state(new_parent_cells, new_ref_pos)
            
            # Change dt to match the step consumed by each particle
            # self.stepper.dt_fn.dat.zero()
            # self.stepper.dt_fn.dat.data_wo[:] = dt_consumed_arr
            # self.stepper.finalize_step()

            self.dt_remaining.dat.data_wo[:] = dt_remaining_arr

            # XXX: moved rebuild from end of outer loop iteration.
            needs_rebuild = self.particle_vom.comm.allreduce(len(boundary_particles_current) + len(migrating_particles_current)>0, op=MPI.LOR)
            if needs_rebuild:
                # record the global IDs of particles about to be consumed whilst PID is still in current ordering
                self.absorbed_pids.extend(self.particle_vom._particle_ids.dat.data_ro[boundary_particles_current])

                new_phys_coords = assemble(
                    interpolate(
                        SpatialCoordinate(self.parent_mesh), self.particle_vom.coordinates.function_space()
                    )
                )

                # Update the VOM's physical coordinates
                self.particle_vom.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro
 
                self.particle_vom_updater.rebuild_vom(
                    absorbed_vom_indices=boundary_particles_current, 
                )

                topo._migrate_functions()

                print(f"[rank {self.particle_vom.comm.rank}] post-rebuild: "
                    f"swarm local size={topo.topology_dm.getLocalSize()} "
                    f"entity_classes={topo._entity_classes} "
                    f"cell_set.size={self.particle_vom.cell_set.size} "
                    f"dt_remaining={self.dt_remaining.dat.data_ro.shape} "
                    f"X={self.particle_vom.reference_coordinates.dat.data_ro.shape}", flush=True)
                
                self.stepper.invalidate()

                # Reconsruct loop state in the new ordering
                # clear/refresh the local-sized bookkeeping arrays
                boundary_particles_current = []
                migrating_particles_current = []

                new_ref_pos = self.particle_vom.reference_coordinates.dat.data_ro.copy()

        else:
            particles_not_resolved_idxs = np.where(dt_remaining_arr > self.eff_time_tol)[0]
            self.logger.print_particles("Non-resolved particles",
                                        {
                                            "dt_remaining": dt_remaining_arr[particles_not_resolved_idxs]
                                        },
                                        indices=particles_not_resolved_idxs,
                                        level="info")
            raise ParticleCrossingLoopNotConverged(
                    f"Cell crossings could not be resolved within {self._params.max_iters} iterations."
                )