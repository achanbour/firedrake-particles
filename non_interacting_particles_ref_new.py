from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater
from non_interacting_particles_phys import move_particles_in_phys_space
from ufl.differentiation import ReferenceGrad

np.random.seed(42)

t = 0.0
dt = 0.1
T = 1

def move_particles_in_ref_space(pmesh, mesh, v_fn, dt, T, t=0.0):
    """
    Update particle positions in reference space using Forward Euler:

    X(t + dt) = X(t) + J^-1*v*dt 
    
    where J = dF/dX is the Jacobian of the map from ref space to physical 
    space F: X -> x.
    """
    # Initialize static variables
    N = pmesh.num_vertices()
    x = SpatialCoordinate(mesh)
    invJ_expr = inv(ReferenceGrad(x))
    ref_cell = mesh.coordinates.function_space().finat_element.cell
    ref_cell_edges = ref_cell.get_topology()[1]
    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    while t < T:
        # At the start of each outer time step, the particle VOM will have been updated.
        # If the update doesn't change the VOM object, the following lines can be moved outside the while loop
        # so that we don't need to re-define FunctionSpaces and Functions every time.
        TFS_vom = TensorFunctionSpace(pmesh, "DG", 0) # Tensor FS for the Jacobian inverse
        invJ_vom = Function(TFS_vom)
        invJ_vom.interpolate(invJ_expr)

        FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for time steps

        # Get the current reference coordinates Function
        ref_coords_fn = pmesh.reference_coordinates

        # Per-particle tracking loop variables
        dt_left = np.full(N, dt) # remaining time for the current time step

        # Run outer loop while there are active particles (those that have not yet finished their dt)
        outer_loop_iter = 0
        while True:
            ref_coords_register = ref_coords_fn.dat.data_ro.copy() # copy of reference coords for book-keeping throughout loops

            active = dt_left > 0
            if not np.any(active):
                break
            
            outer_loop_iter += 1
            active_indices = np.where(active)[0]

            # -- Phase 0 --
            # For all currently active particles, compute updated positions and detect crossings
            # TODO: Check that the assignment of dt_left to dt_trial_fn is correct (agrees on data ordering?)
            dt_trial_fn = Function(FS_vom)
            dt_trial_fn.dat.data[active_indices] = dt_left[active_indices]
            trial_ref_pos_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_trial_fn)

            # Compute barycentric coordinates
            bary_old = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_fn.dat.data_ro))
            bary_new = ref_cell.compute_barycentric_coordinates(np.array(trial_ref_pos_fn.dat.data_ro))
            
            # Detect crossings and split particles into active/passed subsets
            # NOTE: In phase 0, there could be more than one crossing per particle (e.g., when a particle is arbitrarily close to a vertex)
            # which is why we need to implement a separate loop for failed particles to detect the correct crossing
            # (i.e., the crossing that brings the particle to the facet where it succeeds the barycentric test).

            passed_local, t_cross, crossed_edges = detect_crossings(
                bary_old, bary_new, dt_left[active_indices], ref_cell_edges
            )

            passed_global = active_indices[passed_local]
            failed_global = active_indices[~passed_local]
        
            """
            Split active particles into passed/failed groups.

            1. Passed particles (still in cell)
                - set dt_left to 0
                - save ref. coords

            2. Failed particles (left cell)
                - advance position to the *first* crossing facet
                - update dt_left by subtracting t_cross
                - update parent cell to neighbor across crossed facet
                - re-enter outer loop as active with updated ref. pos., parent cell and dt_left
            """
            # -- Process passed particles --
            dt_left[passed_global] = 0
            ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

            print("\n=== POST PHASE 0 ===")
            print("Outer loop iteration:", outer_loop_iter)
            print(f"\nActive particles: {active_indices}")
            print(f"  Failed set: {failed_global}")
            print(f"  Passed set: {passed_global}")

            if len(passed_global) > 0:
                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            print("\nFailed set info:")
            print(f"  dt_left: {dt_left[failed_global]}")

            # -- Process failed particles -> Phase 1 -- 
            # Process active-failed particles though a nested loop that moves them to the correct crossing facet
            # turning them into active-passed particles.
            t_cross_failed = t_cross[~passed_local]
            crossed_edges_failed = crossed_edges[~passed_local]
            if len(failed_global) > 0:
                print("\nGoing into phase 1 to move failed particles to crossing facet...")
                t_cross_final, crossed_edges_final, ref_coords_final = \
                    move_failed_particles_to_facet(
                    failed_global, t_cross_failed, crossed_edges_failed,
                    ref_coords_fn, invJ_vom, v_fn, # ref_coords_fn stores the ref coords of the current VOM
                    FS_vom, ref_cell, ref_cell_edges
                )
                # Update global variables for the failed particles
                # the return variables are only non-zero for the failed particles so passed particles are unaffected.
                dt_left -= t_cross_final
                crossed_edges = crossed_edges_final
                ref_coords_register = ref_coords_final

                print("\n=== POST PHASE 1 ===")
                print("Updated failed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref_coords: {ref_coords_register[failed_global]}")

            else:
                print("Skipping phase 1 as there are no failed particles.")

            # At this point all the particles are now active-passed particles
            # TODO
            # 1) After updating all ref coords, run barycentric test again to ensure all particles are within their cells
            bary_register = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_register))
            tol = 1e-12
            for i, coords_i in enumerate(bary_register):
                if np.any(coords_i < -tol):
                    print(f"Error: Particle {i} is still outside its parent cell after the phase 1 update.")
                    break
            else:
                print("\nPassed barycentric test: all particles are now within their parent cells after the phase 1 update.")
            
            breakpoint()
            # 2) Move particles to neighbouring cells using the crossed_edges info
            # How much of the VOM needs to be updated here?
            # - modify parent cell information
            # update the reference coordinates Function (otherwise the next assemble/interpolate update will give wrong results)
            # - recompute inverse Jacobian using new parent cell ownerhsip
            # 4) Re-enter the outer loop with new ref. coords., parent cells and remaining dt_left
    
        # -- Compute new physical coordinates and update VOM
        new_phys_coords = np.einsum('in, ing->ig', new_bary_coords, cells_coords) # in VOM ordering
        # print("new_phys_coords: ", new_phys_coords)
        new_phys_coords_func = Function(coords.function_space())
        new_phys_coords_func.dat.data[:] = new_phys_coords
        pmesh_updater.update(new_phys_coords_func)

        t += dt

    return t

def update_ref_pos(ref_pos_fn, invJ_vom, v_fn, dt_fn):
    """
    Update particle positions in reference space.

    X(t + dt) = X(t) + J^-1*v*dt

    To distinguish between active and inactive particles within the inner loop,
    pass dt_fn as a scalar DG0 Function giving per-particle time steps.
    """
    # Mesh consistency checks
    m = ref_pos_fn.function_space().mesh()
    assert invJ_vom.function_space().mesh() == m
    assert v_fn.function_space().mesh() == m
    assert dt_fn.function_space().mesh() == m
    
    update_expr = ref_pos_fn + invJ_vom * v_fn * dt_fn
    new_ref_pos_fn = assemble(interpolate(update_expr, ref_pos_fn.function_space()))
    return new_ref_pos_fn

def detect_crossings(bary_old, bary_new, dt_left, edges, tol=1e-12):
    """Detect which particles have crossed cell boundaries based on barycentric coordinates.

    Returns:
    - passed: boolean array indicating which particles passed (i.e., remained inside their cells at their new positions)
    - t_cross: array of crossing times for each particle that failed
    - crossed_edges: list of local IDs of crossed edges for each particle that failed
    """
    n_particles, n_verts = bary_new.shape
    passed = np.ones(n_particles, dtype=bool)
    t_cross = np.full(n_particles, np.inf)
    crossed_edges = [None] * n_particles

    for i, coords_i in enumerate(bary_new):
        # Find all negative barycentric coords in the new position
        negative_verts = np.where(coords_i < -tol)[0]
        if len(negative_verts) == 0:
            # Particle stayed inside its cell
            continue    

        # Mark the particle as having crossed the cell boundary
        passed[i] = False

        if len(negative_verts) == 1:
            # Single deterministic crossing
            neg_vert = negative_verts[0]
        else:
            # Ambiguous crossing
            # Choose the edge with the most negative barycentric coord
            neg_vert = negative_verts[np.argmin(coords_i[negative_verts])]
        
        # Find the local edge ID opposite the negative vertex
        for edge_id, edge_verts in edges.items():
            if neg_vert not in edge_verts:
                crossed_edges[i] = edge_id
                # Compute the intersection time
                old_lamda = bary_old[i, neg_vert]
                new_lambda = coords_i[neg_vert]
                t_cross[i] = -old_lamda / (new_lambda - old_lamda) * dt_left[i]

    crossed_edges = np.array(crossed_edges, dtype=object) # convert to numpy array for easier indexing
    return passed, t_cross, crossed_edges

def move_failed_particles_to_facet(
        failed_global, t_cross0, crossed_edges0, ref_coords_fn, invJ_vom, v_fn, FS_vom, ref_cell, ref_cell_edges, tol=1e-12):
    """Phase 1: Move failed particles to their first crossing facet.
    
    This resolves the case where the phase 0 check produces two negative barycentric
    coordinates indicating that the full step has overshot a vertex. We therefore backtrack
    iteratively with reduced dt until exactly one barycentric coordinate becomes 0.
    """
    n_failed = len(failed_global)
    if n_failed == 0:
        return

    # Declare variables that will be updated in the inner loop
    dt_curr = t_cross0
    still_failed = np.asarray(failed_global, dtype=int)
    edge_ids_curr = np.asarray(crossed_edges0, dtype=object).copy() # This tracks which edge each still_failed particle is associated with

    # Declare end results arrays
    N = FS_vom.mesh().num_vertices()
    t_cross_final = np.zeros(N)
    crossed_edges_final = np.full(N, None, dtype=object)
    ref_coords_final = np.zeros((N, FS_vom.mesh().geometric_dimension))

    # Loop until every failed particle finds a unique crossing facet
    inner_loop_iter = 0
    while len(still_failed) > 0:
        inner_loop_iter += 1
        # NOTE: the below steps replicate what's being done in the outer loop, but only for the currently failed particles set
        # TODO: Check that the assignment of dt_curr to dt_step_fn is correct (agrees on data ordering?)
        dt_step_fn = Function(FS_vom)
        dt_step_fn.dat.data[still_failed] = dt_curr

        ref_step_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_step_fn)

        bary_old = ref_cell.compute_barycentric_coordinates(
            np.array(ref_coords_fn.dat.data_ro)[still_failed]
        )
        bary_new = ref_cell.compute_barycentric_coordinates(
            np.array(ref_step_fn.dat.data_ro)[still_failed]
        )
        passed_mask, t_cross, crossed_edges = detect_crossings(
            bary_old, bary_new, dt_curr, ref_cell_edges
        )

        passed_idx = still_failed[passed_mask]
        failed_idx = still_failed[~passed_mask]

        # Record values for newly passed particles
        t_cross_final[passed_idx] = dt_curr[passed_mask]
        crossed_edges_final[passed_idx] = edge_ids_curr[passed_mask]
        ref_coords_final[passed_idx] = ref_step_fn.dat.data_ro[passed_idx]

        # Update state for next iteration
        still_failed = failed_idx
        dt_curr = t_cross[~passed_mask]
        edge_ids_curr = crossed_edges[~passed_mask]
    
    # Finally, update global variables for the failed particles (all others are zeroed-out by default).
    # dt_left is equal to the original dt_left minus t_cross_inner that made the particle succeed.
    # dt_left -= t_cross_final
    # crossed_edges[:] = crossed_edges_final  
    # ref_coords[:] = ref_coords_final
    
    return t_cross_final, crossed_edges_final, ref_coords_final


if __name__=='__main__':
    # Define the parent mesh
    mesh = UnitSquareMesh(10, 10)

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

    # Update regime 1: Move particles in ref. space
    T_final = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0)
    print("Final particle positions in physical space updated in ref space (in updated VOM order): ", particle_vom.coordinates.dat.data)
