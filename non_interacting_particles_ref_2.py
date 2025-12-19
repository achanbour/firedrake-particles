from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater
from non_interacting_particles_phys import move_particles_in_phys_space
from particle_tracking.topology import find_next_cell
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
        FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for time steps

        # Get the current reference coordinates Function
        ref_coords_fn = pmesh.reference_coordinates

        boundary_particles = [] # list to keep track of particles that hit domain boundaries

        # Per-particle tracking loop variables
        dt_left = np.full(N, dt) # remaining time for the current time step
        ref_coords_register = ref_coords_fn.dat.data_ro.copy() # register to hold updated ref. coords

        # Run outer loop while there are active particles (those that have not yet finished their dt)
        outer_loop_iter = 0
        active_iters = np.zeros(N, dtype=int)
        # breakpoint()
        while True:
            # Sync the ref_coords_register with the current ref_coords_fn data
            if not np.array_equal(ref_coords_register, ref_coords_fn.dat.data_ro):
                ref_coords_register = ref_coords_fn.dat.data_ro.copy()

            active = dt_left > 0
            if not np.any(active):
                break

            outer_loop_iter += 1
            active_indices = np.where(active)[0]
            active_iters[active_indices] += 1

            # -- Phase 0: Process active particles --
            # For all currently active particles, compute updated positions and detect crossings
            dt_trial_fn = Function(FS_vom)

            # NOTE: this line enforces that dt_left indexes into the VOM ordering i.e., dt_left[i] is VOM particle i
            dt_trial_fn.dat.data[active_indices] = dt_left[active_indices]
            invJ_vom.interpolate(invJ_expr) # (re)compute invJ on the CURRENT embedding
            trial_ref_pos_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_trial_fn)

            # Compute barycentric coordinates (on all particles)
            bary_old = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_fn.dat.data_ro))
            bary_new = ref_cell.compute_barycentric_coordinates(np.array(trial_ref_pos_fn.dat.data_ro))
            
            # Detect crossings and split particles into passed/failed subsets (on the currently active set)
            passed_mask, t_cross_local, crossed_edges_local = detect_crossings_linear(
                bary_old[active_indices], bary_new[active_indices], dt_left[active_indices], ref_cell_edges
            )

            # Get local indices (within active set) of passed/failed particles
            passed_local = np.where(passed_mask)[0]
            failed_local = np.where(~passed_mask)[0]

            # Map to global indices (within full particle array)
            passed_global = active_indices[passed_local]
            failed_global = active_indices[failed_local]
        
            """
            Split active particles into passed/failed groups.

            1. Passed particles (still in cell)
                - set dt_left to 0
                - save ref. coords

            2. Failed particles (left the cell)
                - advance position to the facet crossed
                - update dt_left by subtracting t_cross
                - update parent cell to neighbor across crossed facet
                - re-enter outer loop as active with updated ref. pos., parent cell and dt_left
            """
            print(f"\n---Outer loop iteration: {outer_loop_iter}---")
            print(f"Active particles: {active_indices}")
            print(f"  Failed set: {failed_global}")
            print(f"  Passed set: {passed_global}")

            # -- Phase 1: Process passed and failed particles separately --
            # Passed particles
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            # Failed particles
            if len(failed_global) > 0:
                # Get particle positions at crossing facet
                new_ref_coords_failed = move_failed_particles_to_facet(
                    failed_global, t_cross_local[failed_local], ref_coords_fn, invJ_vom, v_fn, FS_vom
                )
                dt_left[failed_global] -= t_cross_local[failed_local]
                ref_coords_register[failed_global] = new_ref_coords_failed

                print("\nFailed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref_coords: {ref_coords_register[failed_global]}")

            # Validate the new particle positions:
            # - Passed particles should be inside their original parent cells
            # - Failed particles should be on a facet of their original parent cells
            #   (i.e., at least one barycentric coordinate is approx 0, none
            print("\nChecking whether all particles are still within their original cells...\n")
            bary_register = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_register))
            tol = 1e-12
            for global_i in passed_global:
                if np.any(bary_register[global_i] < -tol):
                    print(f"Error: Passed particle {global_i} is outside its cell.")
            
            for global_i in failed_global:
                if not np.any(np.abs(bary_register[global_i]) < tol):
                    print(f"Warning: Failed particle {global_i} is not on a facet.")
                if np.any(bary_register[global_i] < -tol):
                    print(f"Error: Failed particle {global_i} has been moved past the facet.")

            print("Barycentric validation complete.")
            
            # 2) Move failed particles to neighbouring cells using the crossed_edges info
            # Get the current parent cell ownership
            # NOTE: update_ref_view deletes the cached properties of the particle VOM based on parent cell ownership 
            # therefore we need to re-access the cell_parent_cell_list attribute after each update_ref_view call 
            # to ensure it gets recomputed correctly.
            print("\nAttempting to search next parent cells...\n")
            parent_cells = pmesh.topology.cell_parent_cell_list # ID of parent cell for each point in VOM order
            next_parent_cells = parent_cells.copy()

            for local_i, global_i in zip(failed_local, failed_global):
                parent_cell = parent_cells[global_i, 0]
                edge_id = crossed_edges_local[local_i]

                next_cell = find_next_cell(mesh, parent_cell, edge_id)

                if next_cell is None:
                    # Exterior boundary hit
                    next_parent_cells[global_i] = parent_cell
                    boundary_particles.append(global_i)
                    dt_left[global_i] = 0.0
                    print(f"Warning: Particle {global_i} attempted to cross an exterior boundary facet from cell {parent_cell}")
                else:
                    next_parent_cells[global_i] = next_cell
            print("\nAll neighbouring cells determined.")

            # - modify parent cell ownership in VOM
            # - update the reference coordinates Function (otherwise the next assemble/interpolate update will give wrong results)
            pmesh_updater.update_ref_view(next_parent_cells, ref_coords_register)

            # - recompute inverse Jacobian using new parent cell ownership (done at start of outer loop)
            # 4) Re-enter the outer loop with new ref. coords., parent cells and remaining dt_left

        print()
        print("=" * 60)
        print("End of time step summary")
        print("-" * 60)
        print(f"  Inner iterations to complete dt        : {outer_loop_iter}")
        print(f"  Active iterations per particle         : {active_iters}")
        print(f"  Boundary particles encountered         : {boundary_particles}")
        print("=" * 60)
        print()

        breakpoint()
        # TODO: Update the VOM by removing all boundary particles
        # i.e., particles that have hit an exterior boundary in one of the iterations above.
        # This causes the VOM topology to change.

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

def detect_crossings_linear(bary_old, bary_new, dt_left, edges, tol=1e-12):
    """Find the facet that each particle crosses (i.e., exits the cell through) in the given time step.
    
    For each particle, compute the time interval on which all barycentric coordinates are non negative. 
    The exit time is taken as the upper end of this interval (corresponding to the last crossing).

    Inputs are assumed to be restricted to the currently active particles only.
    """

    N_active, n_verts = bary_new.shape
    passed = np.ones(N_active, dtype=bool)
    t_cross = np.full(N_active, np.inf)
    crossed_edges = np.full(N_active, None, dtype=object)

    for i in range(N_active):
        dt = dt_left[i]
        lambda_old = bary_old[i]
        lambda_new = bary_new[i]
        dlambda = (lambda_new - lambda_old) / dt # derivative of the barycentric trajectory

        # Time interval during which the particle is inside the cell
        t_in = 0.0
        t_out = dt
        exit_vert = None

        feasible = True

        # The code below seeks the feasible time interval [t_in, t_out] during which all barycentric coordinates are non-negative
        for j in range(n_verts):
            if abs(dlambda[j]) < tol:
                # lambda_j constant in time
                if lambda_old[j] < -tol:
                    feasible = False # particle is outside the cell for the whole time step
                    break
                continue

            t_zero = -lambda_old[j] / dlambda[j] # time when lambda_j crosses zero

            if dlambda[j] > 0:
                # lambda_j increasing: particle enters the cell at t_zero
                if t_zero > t_in:
                    t_in = t_zero
            else:
                # lambda_j decreasing: particle exits the cell at t_zero
                if t_zero < t_out:
                    t_out = t_zero
                    exit_vert = j

        # No valid intersection with the cell
        # NOTE: a slightly negative t_out due to numerical roundoff can cause t_in > t_out so we allow a small tolerance here.
        if not feasible or t_out < -tol or t_out > dt + tol or t_in > t_out + tol:
            passed[i] = False
            t_cross[i] = 0.0
            crossed_edges[i] = None
            continue
        
        # Particle remains inside cell for whole time step
        if t_out >= dt - tol:
            continue

        # Otherwise, particle exits at t_out
        # NOTE: this includes the case where t_out is approx. 0 such as when a particle starts on a facet 
        # and moves outward through the same facet.
        
        passed[i] = False
        t_cross[i] = t_out

        # Exit facet is opposite exit_vert
        for edge_id, edge_verts in edges.items():
            if exit_vert not in edge_verts:
                crossed_edges[i] = edge_id
                break

    return passed, t_cross, crossed_edges
        

def move_failed_particles_to_facet( failed_global, t_cross, ref_coords_fn, invJ_vom,v_fn, FS_vom):
    """Move failed particles exactly to their crossing facet."""

    dt_step_fn = Function(FS_vom)
    dt_step_fn.dat.data[failed_global] = t_cross # set dt to t_cross for failed particles only

    # One single update to the crossing facet
    ref_step_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_step_fn)

    # Extract results for failed particles only
    ref_coords_final = ref_step_fn.dat.data_ro[failed_global].copy()

    return ref_coords_final

if __name__=='__main__':
    # Define the parent mesh
    mesh = UnitSquareMesh(10, 10)
    # mesh = PeriodicUnitSquareMesh(10, 10)

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
    T_final = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0)
    print("Final particle positions in physical space updated in ref space (in updated VOM order): ", particle_vom.coordinates.dat.data)
