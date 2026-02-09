from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater
from particle_tracking.cell_crossing_utils import find_next_cell, compute_ref_coords_in_new_cell
from ufl.differentiation import ReferenceGrad
from FIAT.reference_element import UFCInterval, UFCQuadrilateral, UFCTriangle, UFCHexahedron, UFCTetrahedron

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
    ref_cell_edges = ref_cell.get_topology()[1]

    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    while t < T:
        TFS_vom = TensorFunctionSpace(pmesh, "DG", 0) # Tensor FS for the Jacobian inverse
        invJ_vom = Function(TFS_vom)  
        FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for per-particle time steps

        # Get the current reference coordinates
        ref_coords_fn = pmesh.reference_coordinates

        boundary_particles = [] # list to keep track of particles that hit the domain boundary

        # Define per-particle tracking loop variables
        dt_left = np.full(N, dt) # remaining time for the current time step
        ref_coords_register = ref_coords_fn.dat.data_ro.copy() # registery of updated ref. coords

        # Run inner loop while there are active particles (those that have not yet finished their dt)
        inner_loop_iter = 0
        active_iters = np.zeros(N, dtype=int)

        while True:
            # Ensure ref_coords_register is equal to the "latest" ref coords
            if not np.array_equal(ref_coords_register, ref_coords_fn.dat.data_ro):
                ref_coords_register = ref_coords_fn.dat.data_ro.copy()

            # Check if there any active particles left
            active = dt_left > 0
            if not np.any(active):
                break

            inner_loop_iter += 1
            active_indices = np.where(active)[0]
            active_iters[active_indices] += 1

            # Process active particles
            # For all currently active particles, compute updated positions and detect crossings
            dt_trial_fn = Function(FS_vom)

            # NOTE: this line enforces that dt_left indexes particles in VOM ordering 
            # i.e., dt_left[i] corresponds to VOM particle i
            dt_trial_fn.dat.data[active_indices] = dt_left[active_indices]
            invJ_vom.interpolate(invJ_expr) # recompute invJ on the CURRENT embedding
            trial_ref_pos_fn = advance_ref_coords_euler(ref_coords_fn, invJ_vom, v_fn, dt_trial_fn)

            # Compute barycentric coordinates at the old and new points
            bary_old = ref_cell.compute_barycentric_coordinates(ref_coords_fn.dat.data_ro)
            bary_new = ref_cell.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro)

            # Split particles into passed/failed sets
            passed_mask = np.array([is_inside_cell(bary, tol=1e-12) for bary in bary_new[active_indices]])
            failed_mask = ~passed_mask

            passed_local = np.where(passed_mask)[0] # local indices in the active set
            failed_local = np.where(failed_mask)[0]
            passed_global = active_indices[passed_local] # global indices in the full particle set
            failed_global = active_indices[failed_local]

            # Detect crossings for failed particles
            t_cross = np.full(len(active_indices), np.nan)
            bary_cross = np.full(len(active_indices), None, dtype=object)

            use_bisection = True

            # Option 1: Run an intersection-based algorithm
            t_cross_linear = np.full(len(active_indices), np.nan)
            bary_cross_linear = np.full(len(active_indices), None, dtype=object)

            for local_i, global_i in zip(failed_local, failed_global):
                t_cross_i, bary_cross_i = intersect_crossing_time(
                    bary_old[global_i],
                    bary_new[global_i],
                    dt_left[global_i]
                )
                t_cross_linear[local_i] = t_cross_i
                bary_cross_linear[local_i] = bary_cross_i
            
            # Option 2: Run boolean bisection
            t_cross_bisect = np.full(len(active_indices), np.nan)
            bary_cross_bisect = np.full(len(active_indices), None, dtype=object)

            for local_i, global_i in zip(failed_local, failed_global):
                t_cross_i, X_cross, bary_cross_i = bisect_crossing_time(
                    ref_coords_fn.dat.data_ro[global_i],
                    trial_ref_pos_fn.dat.data_ro[global_i],
                    bary_old[global_i],
                    bary_new[global_i],
                    dt_left[global_i],
                    ref_cell
                )
                t_cross_bisect[local_i] = t_cross_i
                bary_cross_bisect[local_i] = bary_cross_i

            # breakpoint()
            # Check that both methods return the same crossing times and barycentric coordinates
            if use_bisection:
                t_cross = t_cross_bisect
                bary_cross = bary_cross_bisect
            else:
                t_cross = t_cross_linear
                bary_cross = bary_cross_linear
            
            # From the barycentric coords. at the crossing point, deduce which edge the particle crossed
            crossed_edges = np.full(len(active_indices), None, dtype=object)
            for local_i in failed_local:
                crossed_edges[local_i] = int(np.argmin(np.abs(bary_cross[local_i])))
            
            # breakpoint()
            # Check crossed edges
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

            # Process passed and failed particles separately
            # Passed particles
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            # Failed particles
            if len(failed_global) > 0:
                # Compute particle positions at the crossing facet
                # This is done by running the integrator with time step = t_cross 
                crossing_dt_fn = Function(FS_vom)
                crossing_dt_fn.dat.data[failed_global] = t_cross[failed_local]
                ref_step_fn = advance_ref_coords_euler(ref_coords_fn, invJ_vom, v_fn, crossing_dt_fn)

                dt_left[failed_global] -= t_cross[failed_local]
                ref_coords_register[failed_global] = ref_step_fn.dat.data_ro[failed_global]

                print("Failed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref coords (in current cell): {ref_coords_register[failed_global]}")

                # Identify the next cells to move the particles to given the crossed facets
                parent_cells = pmesh.topology.cell_parent_cell_list # parent cell ID for each point in VOM order
                new_parent_cells = parent_cells.copy()

                for local_i, global_i in zip(failed_local, failed_global):
                    parent_cell = parent_cells[global_i, 0]
                    crossed_edge_id = crossed_edges[local_i]

                    next_cell = find_next_cell(mesh, parent_cell, crossed_edge_id)

                    if next_cell is None:
                        # Exterior boundary hit
                        new_parent_cells[global_i] = parent_cell
                        boundary_particles.append(global_i)
                        dt_left[global_i] = 0.0
                        print(f"Warning: Particle {global_i} attempted to cross an exterior boundary facet from cell {parent_cell}")
                    else:
                        new_parent_cells[global_i] = next_cell

                # Compute reference coordinates in the new parent cells
                # TODO: Remove this step by pre-computing the coordinate transforms for all pairs of cells
                # For each cell, precompute neighbouring cell store in an integer field of size num_facets
                # pre compute coordinate transforms (A,b) in a matrix field
                new_ref_coords_in_new_cells = compute_ref_coords_in_new_cell(
                    failed_global,
                    parent_cells,
                    new_parent_cells,
                    crossed_edges[failed_local],
                    ref_coords_register,
                    mesh,
                    ref_cell
                )
                ref_coords_register[failed_global] = new_ref_coords_in_new_cells
                print(f"  new ref coords (in next cells): {ref_coords_register[failed_global]}")

            # 4) Update the particle VOM:
            # - modify parent cell ownership
            # - update the reference coordinates (otherwise the next assemble/interpolate update will give wrong results)
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

        breakpoint()
        # Now update the VOM by removing all boundary particles
        # i.e., particles that have hit an exterior boundary in one of the iterations above.
        # This causes the VOM topology to change.

        # Rebuild the VOM given the updated particle positions
        # NOTE: Physical coordinates can be obtained from reference coordinates by interpolating the parent mesh into the VOM
        # Interpolation makes use of the parent cell ownership information and reference coordinates of VOM points
        # new_phys_coords = compute_phys_coords_from_ref_coords(ref_coords_register, pmesh.topology.cell_parent_cell_list, mesh, ref_cell)
        new_phys_coords = assemble(interpolate(SpatialCoordinate(mesh), pmesh.coordinates.function_space()))
        pmesh_updater.rebuild_vom(absorbed_vom_indices=boundary_particles, new_coords=new_phys_coords.dat.data_ro)

        t += dt

    return t

def advance_ref_coords_euler(ref_pos_fn, invJ_vom, v_fn, dt_fn):
    """
    Advance particle forward by one Euler step in reference space.

    X(t + dt) = X(t) + J^-1*v*dt

    To distinguish between active and inactive particles within the inner loop,
    pass dt_fn as a scalar DG0 Function storing per-particle time steps.
    """
    # Mesh consistency checks
    m = ref_pos_fn.function_space().mesh()
    assert invJ_vom.function_space().mesh() == m
    assert v_fn.function_space().mesh() == m
    assert dt_fn.function_space().mesh() == m
    
    update_expr = ref_pos_fn + invJ_vom * v_fn * dt_fn
    new_ref_pos_fn = assemble(interpolate(update_expr, ref_pos_fn.function_space()))
    return new_ref_pos_fn

def intersect_crossing_time(bary0, bary1, dt, tol=1e-12):
    """
    Detect the facet crossed and the crossing time for a single particle, assuming a linear barycentric trajectory over [0, dt].
    This works for linear particle trajectories and affine meshes
    """

    n_coords = len(bary0)
    dlambda = (bary1 - bary0) / dt

    t_out = dt
    exit_coord = None

    # Preconditions
    assert is_inside_cell(bary0, tol=tol) # old point lies inside the cell
    assert not is_inside_cell(bary1, tol=tol) # new point lies outside the cell

    for j in range(n_coords):
        if abs(dlambda[j]) < tol:
            # Constant barycentric coordinate
            if bary0[j] < -tol:
                raise RuntimeError(
                    "Particle starts outside cell in linear crossing detection."
                )
            continue

        # Time when lambda_j(t) = 0
        t_zero = -bary0[j] / dlambda[j]

        # Only decreasing coordinates can cause exit
        if dlambda[j] < 0 and -tol <= t_zero <= t_out + tol:
            if t_zero < t_out:
                t_out = t_zero
                exit_coord = j

    if exit_coord is None or t_out < -tol or t_out > dt + tol:
        raise RuntimeError(
            "Linear crossing detection failed: no valid exit found."
        )
    
    t_cross = max(0.0, min(t_out, dt))
    bary_cross = bary0 + t_cross * dlambda

    return t_cross, bary_cross
        
# New: implement bisection to detect crossed edge and crossing time
def is_inside_cell(bary, tol):
    """Boolean predicate: is particle inside the cell?"""
    return np.all(bary >= -tol)

def X_at_t(X0, X1, t, dt):
    """
    This function returns the reference coordinates of a single particle
    within a single forward Euler update step.

    NOTE: for an arbitrary integrator, this needs to be replaced by calling the integrator directly.
    calling `advance_ref_coords_euler` is expensive and it's a global operation involving the whole VOM.
    """
    return X0 + (t / dt) * (X1 - X0)

# TODO: Rewrite bisection as a SIMD operation over the entire set of points
def bisect_crossing_time(X0, X1, bary0, bary1, dt, ref_cell, tol=1e-12, max_iters=30):
    """Per particle boolean bisection.
    
    Search for the last index the particle satisfies the boolean predicate `is_inside_cell`.
    """

    # Preconditions
    assert is_inside_cell(bary0, tol=tol) # old point lies inside the cell
    assert not is_inside_cell(bary1, tol=tol) # new point lies outside the cell

    t_lo = 0.0
    t_hi = dt

    for _ in range(max_iters):
        t_mid = (t_lo + t_hi) / 2
        X_mid = X_at_t(X0, X1, t_mid, dt)

        bary_mid = ref_cell.compute_barycentric_coordinates(X_mid)

        # Break if the particle hits a boundary
        if np.any(np.abs(bary_mid) < tol):
            t_lo = t_mid
            break

        # Evaluate predicate at the midpoint and update the range
        if is_inside_cell(bary_mid, tol):
            t_lo = t_mid
        else:
            t_hi = t_mid
    
    t_cross = t_lo
    X_cross = X_at_t(X0, X1, t_cross, dt)
    bary_cross = ref_cell.compute_barycentric_coordinates(X_cross)

    return t_cross, X_cross, bary_cross

if __name__=='__main__':
    # Define the parent mesh
    mesh = UnitSquareMesh(10, 10, quadrilateral=False)
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
    print("Final particle positions: ", particle_vom.coordinates.dat.data)
