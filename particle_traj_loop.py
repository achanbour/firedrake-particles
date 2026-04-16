from firedrake import *
from ufl.differentiation import ReferenceGrad
import numpy as np
import warnings
from update_vom import VertexOnlyMeshUpdater, EmptyVOMError
from particle_time_stepper import ForwardEulerTimeStepper
from particle_logger import ParticleLogger
# from plot_vom import plot_particles_snapshot

class ParticleCrossingLoopNotConverged(RuntimeError):
    """Raised when cell crossings within a single time step are not resolved within 
    the maximum number of iterations."""
    pass

def solve_particle_traj_in_ref_space(
        pmesh, mesh, v_fn, dt, T, t=0.0, 
        max_inner_iters=50, 
        max_bisection_iters=None,
        bary_tol=1e-9,
        time_tol=1e-9,
        plot=False,
        log_level="info"):
    """
    Reconstructs the trajectory of particles using a time stepping scheme in reference space.
    """

    logger = ParticleLogger(level=log_level)

    x = SpatialCoordinate(mesh)
    invJ_expr = inv(ReferenceGrad(x))
    ref_cell = mesh.coordinates.function_space().finat_element.cell

    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    # Create reusable function spaces and stepper outside the time loop
    TFS_vom = TensorFunctionSpace(pmesh, "DG", 0) # Tensor FS for the Jacobian inverse
    invJ_vom = Function(TFS_vom)

    FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for per-particle time steps
    dt_trial_fn = Function(FS_vom)

    stepper = ForwardEulerTimeStepper(
        pmesh.reference_coordinates,
        invJ_vom,
        v_fn,
        dt_trial_fn
    )
    
    # Track boundary particles
    particle_ids = np.arange(pmesh.num_vertices())
    removed_particles = []

    # Initialise plot
    if plot:
        import matplotlib.pyplot as plt
        from firedrake.pyplot import triplot, pointplot
        fig, axes = plt.subplots()
        triplot(mesh, axes=axes)
        sc = pointplot(pmesh, axes=axes)
        # prevent matplotlib from rescaling the axes as particles move (making the animation jumpy)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_aspect("equal")
        frame = 0

    outer_time_loop = 0
    while t < T - 1e-12:
        N = pmesh.num_vertices()

        outer_time_loop += 1
        logger.outer_loop(outer_time_loop, t, dt, T, N)

        boundary_particles_current = [] # particles that hit the domain boundary in current time step

        dt_left = np.full(N, dt) # remaining time for the current time step
        ref_coords_register = pmesh.reference_coordinates.dat.data_ro.copy() # array to register updated ref. coords.

        # Run inner loop while there are active particles (those that have not yet finished their dt)
        inner_loop_iter = 0
        active_iters = np.zeros(N, dtype=int)

        while inner_loop_iter < max_inner_iters:            
            # Check if there are any active particles left
            # A particle remains active if it has more than the equivalent of one bisection tol. of remaining time
            active = dt_left > time_tol
            if not np.any(active):
                break

            # Process active particles
            inner_loop_iter += 1
            active_indices = np.where(active)[0]
            active_iters[active_indices] += 1

            stepper.dt.dat.zero()
            stepper.dt.dat.data_wo[active_indices] = dt_left[active_indices]
            
            # Recompute invJ on the CURRENT embedding
            # This is is done here rather than in the outer loop as cell ownership changes within the inner loop
            invJ_vom.interpolate(invJ_expr)
            
            # Get updated reference positions using the full time step
            trial_ref_pos_fn = stepper.step()
            # logger.inspect("trial ref pos", trial_ref_pos_fn.dat.data_ro, level="info")

            # Compute barycentric coordinates at the new positions
            bary_new = ref_cell.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro)

            # Split particles into passed/failed sets
            passed_mask = np.all(bary_new[active_indices] >= -bary_tol, axis=1)
            failed_mask = ~passed_mask

            passed_local = np.where(passed_mask)[0] # local indices in the active set
            failed_local = np.where(failed_mask)[0]
            passed_global = active_indices[passed_local] # global indices in the full particle set
            failed_global = active_indices[failed_local]

            """
            Process passed and failed particles.

            1. Passed particles (still in cell)
                - set dt_left to 0
                - register ref. coords

            2. Failed particles (left the cell)
                - advance position to the crossed facet
                - update dt_left by subtracting t_cross
                - update parent cell to neighbour across the crossed facet
                - re-enter inner loop as active with updated ref. pos., parent cell and dt_left
            """
            logger.inner_loop(inner_loop_iter, active_indices, passed_global, failed_global)

            # Passed particles
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                logger.print_particles("passed particles", {
                    "new_ref_coords": ref_coords_register[passed_global],
                }, indices=passed_global, level="info")

            parent_cells = pmesh.topology.cell_parent_cell_list # parent cell ID for each point in VOM order
            new_parent_cells = parent_cells.copy()

            # Failed particles
            if len(failed_global) > 0:
                """
                For failed particles,
                1. Identify the crossed facet using barycentric coordinates
                2. Determine which cell to go to next
                2. Compute reference coordinates in the new cell.
                """
                if max_bisection_iters is not None:
                    t_cross, bary_cross, X_cross = bisect_crossing_time(stepper, dt_left, ref_cell, failed_global, bary_tol=bary_tol, time_tol=time_tol, max_iters=max_bisection_iters)
                else:
                    t_cross, bary_cross, X_cross = bisect_crossing_time(stepper, dt_left, ref_cell, failed_global, bary_tol=bary_tol, time_tol=time_tol)

                dt_left[failed_global] -= t_cross
                
                logger.print_particles("failed particles", {
                    "dt_left": dt_left[failed_global],
                    "t_cross": t_cross,
                    "bary_cross": bary_cross,
                    "X_cross": X_cross,
                }, indices=failed_global, level="info")
            
                # From the barycentric coords. at the crossing point, determine which edge the particle crossed
                # NOTE: by default, `np.argmin` picks deterministically the edge of the first bary coord that's zero
                # That's fine when bisection finds a crossing point at a vertex but may cause problems if that's the point
                # the particle started on (the particle essentially gets stuck being sent back and forth between the two vertices).
                
                local_crossed_edge_ids = np.full(len(active_indices), None, dtype=object)

                # Separate cases where the crossing is on a vertex or on an edge
                # this introduces a coupling between bary_tol and time_tol which isn't great
                """
                for idx, local_i in enumerate(failed_local):
                    # NOTE: This line will only detect near-zero bary coord. if bary_tol is large enough relative
                    # to the residual from bisection.
                    zero_bary_coord_idx = np.where(abs(bary_cross[idx]) < bary_tol)[0]
                    if len(zero_bary_coord_idx) == 1:
                        # On an edge: 
                        local_crossed_edge_ids[local_i] = int(np.argmin(abs(bary_cross[idx])))
                    else:
                        # On a vertex — two edges are candidates,
                        # pick the edge whose facet normal points in the same direction as the velocity vector
                        crossed_edge = int(zero_bary_coord_idx[0]) # fallback
                        v_ref = invJ_vom.dat.data_ro[failed_global[idx]] @ stepper.v.dat.data_ro[failed_global[idx]]
                        for edge_id in zero_bary_coord_idx:
                            normal = ref_cell.compute_normal(edge_id)
                            if np.dot(normal, v_ref) > 0:
                                crossed_edge = edge_id
                                break

                        local_crossed_edge_ids[local_i] = crossed_edge
                """

                # Alternatively, always check that the selected edge agrees with the direction the velocity points to
                # so the particle never gets stuck.
                for idx, local_i in enumerate(failed_local):
                    crossed_edge = int(np.argmin(abs(bary_cross[idx]))) # starting edge is the one corresponding to the smallest bary coord

                    v_ref = invJ_vom.dat.data_ro[failed_global[idx]] @ stepper.v.dat.data_ro[failed_global[idx]]
                    normal = ref_cell.compute_reference_normal(1, crossed_edge)
                    if np.dot(v_ref, normal) <= 0:
                        # check other edges
                        for edge_id in range(len(bary_cross[idx])):
                            if edge_id == crossed_edge:
                                continue
                            normal = ref_cell.compute_reference_normal(1, edge_id)
                            if np.dot(v_ref, normal) > 0:
                                crossed_edge = edge_id
                                break
                    local_crossed_edge_ids[local_i] = crossed_edge


                # Identify the next cells to move the particles to given the crossed facets
                with PETSc.Log.Event("LookupCellTransitions"):
                    for j, global_i in enumerate(failed_global):
                        # NOTE:
                        # j indexes into the set of failed particles
                        # failed_local[j] gives the index of that particle within the active set
                        # global_i gives the index of that particle in the full set of particles

                        parent_cell = parent_cells[global_i, 0]

                        local_crossed_edge_id = local_crossed_edge_ids[failed_local[j]]

                        next_cell = mesh.topology.cell_facet_neighbours.data[parent_cell, local_crossed_edge_id]

                        if next_cell is None or next_cell == -1:
                            # Exterior boundary hit
                            new_parent_cells[global_i, 0] = parent_cell
                            boundary_particles_current.append(global_i)
                            removed_particles.append(particle_ids[global_i])
                            dt_left[global_i] = 0.0
                            warnings.warn(f"Particle {global_i} attempted to cross an exterior boundary facet from cell {parent_cell}")
                        else:
                            new_parent_cells[global_i, 0] = next_cell

                        # Apply coordinate transforms to get ref. coords in the neighbouring cell
                        # NOTE: This is applied to boundary particles as well with A = 0 and b = 0
                        # but this is fine since these coordinates are never used again.
                        # Boundary particles are no longer considered in any subsequent inner loop iteration (as we set their `dt_left` to 0)
                        # and are immediately discarded once the inner loop terminates.
                        
                        A_facet_coord_transform, b_facet_coord_transform = mesh.topology.cell_facet_coord_transforms
                        ref_coords_register[global_i] = A_facet_coord_transform.data[parent_cell, local_crossed_edge_id] @ X_cross[j] + b_facet_coord_transform.data[parent_cell, local_crossed_edge_id]
                    
                logger.print_particles("failed particles — cell transitions", {
                    "parent_cell": parent_cells[failed_global, 0],
                    "crossed_edge": local_crossed_edge_ids[failed_local],
                    "next_cell": new_parent_cells[failed_global, 0],
                    "new_ref_coords": ref_coords_register[failed_global],
                }, indices=failed_global, level="info")

                # logger.inspect_particles("failed particles", {
                #     "dt left": dt_left[failed_global],
                #     "crossed edges": local_crossed_edge_ids,
                #     "new parent cells": new_parent_cells,
                #     "new ref coords": ref_coords_register[failed_global]
                # }, level="info")

            # 4) Update the particle VOM:
            # - modify parent cell ownership
            # - update the reference coordinates
            pmesh_updater.update_ref_view(new_parent_cells, ref_coords_register)

            # - recompute inverse Jacobian using new parent cell ownership (done at start of inner loop)
            # 5) Re-enter the inner loop with new ref. coords., parent cells and remaining dt_left

            if inner_loop_iter == max_inner_iters:
                still_active = np.where(dt_left > time_tol)[0]
                logger.print_particles(
                    "Non-converged particles",
                    {"dt_left": dt_left[still_active]},
                    indices=still_active,
                    level="info",
                )
                breakpoint()
                raise ParticleCrossingLoopNotConverged(
                    f"Cell crossings were not resolved within {max_inner_iters} iterations."
                )

        # Now update the VOM by removing all boundary particles
        # i.e., particles that have hit an exterior boundary in one of the iterations above.
        # This operation causes the VOM topology to change.
        new_phys_coords = assemble(interpolate(SpatialCoordinate(mesh), pmesh.coordinates.function_space()))

        logger.outer_summary(
            outer_time_loop, inner_loop_iter, active_iters,
            boundary_particles_current,
            pmesh.reference_coordinates.dat.data_ro,
            new_phys_coords.dat.data_ro
        )

        if len(boundary_particles_current) != 0:
            # TODO: Trigger exchange: for each rank constructs 2 sets of particles: absorbed (left mesh domain or crossed partition boundary) + arrived
            pmesh_updater.rebuild_vom(absorbed_vom_indices=boundary_particles_current, new_coords=new_phys_coords)

            # Update/rebuild all fields eagerly (in parallel: once exchange is over)
            # And ensure the stepper stores the updated fields!
            # stepper.X = pmesh.reference_coordinates
            stepper.invalidate()

            stepper.invJ._match_mesh_topology_version()
            stepper.dt._match_mesh_topology_version()
            stepper.v._match_mesh_topology_version()

            # Only retain the ID of surviving particles
            keep_mask = np.ones(len(particle_ids), dtype=bool)
            keep_mask[boundary_particles_current] = False
            particle_ids = particle_ids[keep_mask]

        else:
            # Write physical coordinates back
            # This is simpler than calling pmesh_updater.update_vom()
            pmesh.coordinates.dat.data_wo[:] = new_phys_coords.dat.data_ro

        if plot:
            sc.set_offsets(pmesh.coordinates.dat.data_ro)
            plt.savefig(f"output/frame_{frame:04d}.png", dpi=150)
            frame += 1

        t += dt
    
    if plot:
        plt.close(fig)

    return t, removed_particles

class BisectionNotConvergedError(RuntimeError):
    """Raised when the bisection algorithm has not converged within the maximum number of allowed iterations"""
    pass

BISECTION_COUNT = 0
def bisect_crossing_time(
        stepper,
        dt_left,
        ref_cell, 
        failed_global,
        bary_tol,
        time_tol,
        max_iters=30
):
    """Bisection algorithm that detects particle crossings.
    
    Returns crossing times and reference coordinates at the crossing point for each failed particle.
    """
    global BISECTION_COUNT
    BISECTION_COUNT += 1
    n_failed = len(failed_global)

    # Per particle bisection brakets [t_lo, t_hi]
    t_lo = np.zeros(n_failed, dtype=float)
    t_hi = dt_left[failed_global].copy()

    # NOTE: bisection assumes that initially (at t_lo=0) a particle starts strictly inside its cell
    # This invariant breaks when a particle starts on a facet or vertex or slightly outside (due to floating point noise)
    # However, bisection should still find t_cross > 0 provided the particle is pushed into the cell.

    for _ in range(max_iters):
        t_mid = (t_lo + t_hi) / 2

        stepper.dt.dat.zero()
        stepper.dt.dat.data_wo[failed_global] = t_mid
    
        # Advance only failed particles by mid time substep
        mid_ref_fn = stepper.step()

        X_mid = mid_ref_fn.dat.data[failed_global]
        bary_mid = ref_cell.compute_barycentric_coordinates(X_mid)
        inside = np.all(bary_mid >= -bary_tol, axis = 1)
                
        # For particles inside at the midpoint, advance lower end of the bracket
        t_lo[inside] = t_mid[inside]
        # For particles outside at the midpoint, advance higher end of the bracket
        t_hi[~inside] = t_mid[~inside]
        
        # Early exit if all brackets shrink sufficiently
        if np.max(t_hi - t_lo) < time_tol:
            break
    else:
        raise BisectionNotConvergedError(
            f"Bisection did not converge within {max_iters} iterations."
        )
    
    # Extract crossing times
    t_cross = t_lo

    # Compute barycentric coordinates at the crossing point
    stepper.dt.dat.zero()
    stepper.dt.dat.data_wo[failed_global] = t_cross
    cross_ref_fn = stepper.step()
    X_cross = cross_ref_fn.dat.data_ro[failed_global]
    bary_cross = ref_cell.compute_barycentric_coordinates(X_cross)

    return t_cross, bary_cross, X_cross

# TODO:
# - Check robustness of cell crossing with higher order mesh coordinate field
# - End of particle loop: halo exchange + update all fields eagerly

