from firedrake import *
from firedrake.petsc import PETSc
from ufl.differentiation import ReferenceGrad
import numpy as np
import warnings
from update_vom import VertexOnlyMeshUpdater
from particle_time_stepper import ForwardEulerTimeStepper

np.random.seed(42)

t = 0.0
dt = 0.1
T = 1

def move_particles_in_ref_space(pmesh, mesh, v_fn, dt, T, t=0.0):
    """
    Update particles in reference space using Forward Euler:

    X(t + dt) = X(t) + J^-1*v*dt 
    
    where J = dF/dX is the Jacobian of the geometric map F: X -> x.
    """
    N = pmesh.num_vertices()
    x = SpatialCoordinate(mesh)
    invJ_expr = inv(ReferenceGrad(x))
    ref_cell = mesh.coordinates.function_space().finat_element.cell

    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    # Create reusable function spaces and stepper outside the time loop
    TFS_vom = TensorFunctionSpace(pmesh, "DG", 0) # Tensor FS for the Jacobian inverse
    invJ_vom = Function(TFS_vom)

    FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for per-particle time steps
    dt_trial_fn = Function(FS_vom)

    ref_coords_fn = pmesh.reference_coordinates # current reference coordinates

    stepper = ForwardEulerTimeStepper(
        ref_coords_fn,
        invJ_vom,
        v_fn,
        dt_trial_fn
    )

    while t < T:
        boundary_particles = [] # list to keep track of particles that hit the domain boundary

        # Define per-particle tracking loop variables
        dt_left = np.full(N, dt) # remaining time for the current time step
        ref_coords_register = ref_coords_fn.dat.data_ro.copy() # array to store updating ref. coords.

        # Run inner loop while there are active particles (those that have not yet finished their dt)
        inner_loop_iter = 0
        active_iters = np.zeros(N, dtype=int)

        while True:
            # Ensure ref_coords_register holds the latest ref coords
            if not np.array_equal(ref_coords_register, ref_coords_fn.dat.data_ro):
                ref_coords_register = ref_coords_fn.dat.data_ro.copy()

            # Check if there are any active particles left
            active = dt_left > 0
            if not np.any(active):
                break

            inner_loop_iter += 1
            active_indices = np.where(active)[0]
            active_iters[active_indices] += 1

            # Process active particles
            # NOTE: this line enforces that dt_left indexes particles in VOM ordering 
            # i.e., dt_left[i] corresponds to VOM particle i
            stepper.dt.dat.zero()
            stepper.dt.dat.data_wo[active_indices] = dt_left[active_indices]
            
            # Recompute invJ on the CURRENT embedding
            # NOTE: This is is done here rather than in the outer loop as cell ownership changes
            # between iterations of the inner loop.
            invJ_vom.interpolate(invJ_expr)
            
            trial_ref_pos_fn = stepper.step()

            # Compute barycentric coordinates at the old and new points
            bary_new = ref_cell.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro)

            # Split particles into passed/failed sets
            passed_mask = np.all(bary_new[active_indices] >= -1e-12, axis=1)
            failed_mask = ~passed_mask

            passed_local = np.where(passed_mask)[0] # local indices in the active set
            failed_local = np.where(failed_mask)[0]
            passed_global = active_indices[passed_local] # global indices in the full particle set
            failed_global = active_indices[failed_local]

            """
            Process passed and failed particles.

            1. Passed particles (still in cell)
                - set dt_left to 0
                - save ref. coords

            2. Failed particles (left the cell)
                - advance position to the facet crossed
                - update dt_left by subtracting t_cross
                - update parent cell to neighbour across the crossed facet
                - re-enter inner loop as active with updated ref. pos., parent cell and dt_left
            """
            print(f"\n---Inner loop iteration: {inner_loop_iter}---")
            print(f"Active particles: {active_indices}")
            print(f"  Failed set: {failed_global}")
            print(f"  Passed set: {passed_global}")

            # Passed particles
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            # Failed particles
            if len(failed_global) > 0:
                """
                For failed particles,
                1. Identify the crossed facet using barycentric coordinates
                2. Determine which cell to go to next
                2. Compute reference coordinates in the new cell.
                """
                t_cross, bary_cross, X_cross = bisect_crossing_time_simd(stepper, dt_left, ref_cell, failed_global)

                dt_left[failed_global] -= t_cross
                print("Failed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref coords (in current cell): {X_cross}")
            
                # From the barycentric coords. at the crossing point, determine which edge the particle crossed
                local_crossed_edge_ids = np.full(len(active_indices), None, dtype=object)
                for idx, local_i in enumerate(failed_local):
                    local_crossed_edge_ids[local_i] = int(np.argmin(abs(bary_cross[idx])))

                    # Catch the degenerate case when a particle lands on a vertex
                    # two barycentric coords are 0 so argmin is ambiguous
                    eps = 1e-12
                    near_zero = abs(bary_cross[idx]) < eps
                    if np.count_nonzero(near_zero) >= 2:
                        warnings.warn(
                            f"Degenerate crossing: particle landed on a vertex.\n"
                            f"bary_cross = {bary_cross[idx]}\n"
                            f"failed_global particle = {failed_global[idx]}"
                        )
                        breakpoint()

                # Identify the next cells to move the particles to given the crossed facets
                parent_cells = pmesh.topology.cell_parent_cell_list # parent cell ID for each point in VOM order
                new_parent_cells = parent_cells.copy()

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
                            new_parent_cells[global_i] = parent_cell
                            boundary_particles.append(global_i)
                            dt_left[global_i] = 0.0
                            warnings.warn(f"Particle {global_i} attempted to cross an exterior boundary facet from cell {parent_cell}")
                        else:
                            new_parent_cells[global_i] = next_cell
                        
                        A_facet_coord_transform, b_facet_coord_transform = mesh.topology.cell_facet_coord_transforms
                        ref_coords_register[global_i] = A_facet_coord_transform.data[parent_cell, local_crossed_edge_id] @ X_cross[j] + b_facet_coord_transform.data[parent_cell, local_crossed_edge_id]

                    print(f"  new ref coords (in next cells): {ref_coords_register[failed_global]}")
            
            # 4) Update the particle VOM:
            # - modify parent cell ownership
            # - update the reference coordinates 
            pmesh_updater.update_ref_view(new_parent_cells, ref_coords_register)

            # - recompute inverse Jacobian using new parent cell ownership (done at start of inner loop)
            # 5) Re-enter the inner loop with new ref. coords., parent cells and remaining dt_left

        print()
        print("=" * 60)
        print("End of time step summary")
        print("-" * 60)
        print(f"  Inner iterations to complete dt        : {inner_loop_iter}")
        print(f"  Active iterations per particle         : {active_iters}")
        print(f"  Boundary particles encountered         : {boundary_particles}")
        print("=" * 60)
        print()

        # Now update the VOM by removing all boundary particles
        # i.e., particles that have hit an exterior boundary in one of the iterations above.
        # This operation causes the VOM topology to change.

        # Rebuild the VOM given the updated particle positions
        new_phys_coords = assemble(interpolate(SpatialCoordinate(mesh), pmesh.coordinates.function_space()))
        pmesh_updater.rebuild_vom(absorbed_vom_indices=boundary_particles, new_coords=new_phys_coords.dat.data_ro)

        t += dt

    return t

def bisect_crossing_time_simd(
        stepper,
        dt_left,
        ref_cell, 
        failed_global,
        tol=1e-4,
        max_iters=30
):
    """SIMD-style bisection algorithm that detects particle crossings.
    
    Instead of reconstructing midpoints using a hard-coded linear interpolation,
    this function evaluates midpoint positions by re-running the time stepping routine.

    Returns crossing times and reference coordinates at the crossing point for failed particles.
    """
    n_failed = len(failed_global)

    # Per particle bisection brakets [t_lo, t_hi]
    t_lo = np.zeros(n_failed, dtype=float)
    t_hi = dt_left[failed_global].copy()

    # Define a DG0 timestep Function
    # dt_mid_fn = Function(FS_vom)

    for _ in range(max_iters):
        t_mid = (t_lo + t_hi) / 2
        # dt_mid_fn.dat.data[failed_global] = t_mid

        # Instead of defining a brand-new function update the data the kernel reads
        # This is fine because `dt_trial_fn` gets reset before the next step evaluation
        # stepper.dt.dat.data[:] = 0
        stepper.dt.dat.zero()
        stepper.dt.dat.data_wo[failed_global] = t_mid

        # Advance only failed particles by mid time substep
        mid_ref_fn = stepper.step()

        X_mid = mid_ref_fn.dat.data[failed_global]
        bary_mid = ref_cell.compute_barycentric_coordinates(X_mid)
        inside = np.all(bary_mid >= -tol, axis = 1)
        
        # For particles inside at the midpoint, advance lower end of the bracket
        t_lo[inside] = t_mid[inside]
        # For particels outside at the midpoint, advance higher end of the bracket
        t_hi[~inside] = t_mid[~inside]
        
        # Early exit if all brackets shrink sufficiently
        if np.max(t_hi - t_lo) < tol:
            break
    
    # Extract crossing times
    t_cross = t_lo

    # Compute barycentric coordinates at the crossing point
    stepper.dt.dat.zero()
    stepper.dt.dat.data_wo[failed_global] = t_cross
    cross_ref_fn = stepper.step()
    X_cross = cross_ref_fn.dat.data_ro[failed_global]
    bary_cross = ref_cell.compute_barycentric_coordinates(X_cross)

    return t_cross, bary_cross, X_cross

if __name__=='__main__':
    # Define the parent mesh
    mesh = UnitSquareMesh(10, 10, quadrilateral=False)
    # mesh = PeriodicUnitSquareMesh(10, 10)

    with PETSc.Log.Event("PreComputeCellFacetData"):
        _ = mesh.topology.cell_facet_neighbours
        _ = mesh.topology.cell_facet_coord_transforms

    # Define the particles in a VOM
    N = 10
    particle_coords = np.random.rand(N, 2)
    particle_vom = VertexOnlyMesh(mesh, particle_coords)
    gdim = particle_vom.geometric_dimension
    print("Initial particle positions (in primary VOM order): ", particle_vom.coordinates.dat.data_ro)

    # Assign per-particle velocities
    V = VectorFunctionSpace(particle_vom, "DG", 0, dim=gdim)
    V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=gdim)
    v = Function(V)
    v_io = Function(V_io)
    input_velocities = np.random.normal(0.0, 0.5, size=(N,2))
    v_io.dat.data[:] = input_velocities
    v.interpolate(v_io)

    # Set the parameters below to do a single integration step 
    T = 0.3
    dt = T

    # Move particles in ref. space
    with PETSc.Log.Event("ParticleTrajectoryLoop"):
        T_final = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0)
    print("Final particle positions: ", particle_vom.coordinates.dat.data)

    from particle_time_stepper import STEP_COUNT
    print("Total ForwardEulerTimeStepper step calls: ", STEP_COUNT)

    # from pyop2.caching import print_cache_stats
    # print_cache_stats()
    # replace by: PYOP2_CACHE_INFO=1

    from pyop2.parloop import PARLOOP_CALL_COUNT
    print("Total ParLoop kernel calls: ", PARLOOP_CALL_COUNT)
