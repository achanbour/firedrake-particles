from firedrake import *
import numpy as np
import warnings
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

    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    while t < T:
        TFS_vom = TensorFunctionSpace(pmesh, "DG", 0) # Tensor FS for the Jacobian inverse
        invJ_vom = Function(TFS_vom)  
        FS_vom = FunctionSpace(pmesh, "DG", 0) # Scalar FS for per-particle time steps
        ref_coords_fn = pmesh.reference_coordinates # current reference coordinates

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
            #  bary_old = ref_cell.compute_barycentric_coordinates(ref_coords_fn.dat.data_ro)
            bary_new = ref_cell.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro)

            # Split particles into passed/failed sets
            # passed_mask = np.array([is_inside_cell(bary, tol=1e-12) for bary in bary_new[active_indices]])
            passed_mask = np.all(bary_new[active_indices] >= -1e-12, axis=1)
            failed_mask = ~passed_mask

            passed_local = np.where(passed_mask)[0] # local indices in the active set
            failed_local = np.where(failed_mask)[0]
            passed_global = active_indices[passed_local] # global indices in the full particle set
            failed_global = active_indices[failed_local]

            # Detect crossings for failed particles
            if len(failed_global) > 0:
                t_cross, bary_cross, X_cross = bisect_crossing_time_simd(ref_coords_fn, invJ_vom, v_fn, dt_left, ref_cell, failed_global, FS_vom)
            
                # From the barycentric coords. at the crossing point, determine which edge the particle crossed
                crossed_edges = np.full(len(active_indices), None, dtype=object)
                for idx, local_i in enumerate(failed_local):
                    crossed_edges[local_i] = int(np.argmin(abs(bary_cross[idx])))

                    # Catch degenerate case when a particle lands on a vertex
                    # Two barycentric coords are 0 so argmin is ambiguous
                    eps = 1e-12
                    near_zero = abs(bary_cross[idx]) < eps
                    if np.count_nonzero(near_zero) >= 2:
                        warnings.warn(
                            f"Degenerate crossing: particle landed on a vertex.\n"
                            f"bary_cross = {bary_cross[idx]}\n"
                            f"failed_global particle = {failed_global[idx]}"
                        )
                        breakpoint()

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
                dt_left[failed_global] -= t_cross
                ref_coords_register[failed_global] = X_cross

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

                    """
                    # Convert FIAT facet ID to DMPlex facet point
                    facet_point = mesh.topology.cell_closure[parent_cell][mesh.ufl_cell().num_vertices + crossed_edge_id]

                    # Find local facet index in cone ordering
                    plex_cell = mesh.topology.cell_closure[parent_cell, -1]
                    cone = mesh.topology_dm.getCone(plex_cell)
                    local_facet = None
                    for lf, pt in enumerate(cone):
                        if pt == facet_point:
                            local_facet = lf
                            break
                    
                    next_cell = cell_neighbours[parent_cell, local_facet]
                    """

                    if next_cell is None or next_cell == -1:
                        # Exterior boundary hit
                        new_parent_cells[global_i] = parent_cell
                        boundary_particles.append(global_i)
                        dt_left[global_i] = 0.0
                        warnings.warn(f"Particle {global_i} attempted to cross an exterior boundary facet from cell {parent_cell}")
                    else:
                        new_parent_cells[global_i] = next_cell

                    """
                    # Apply cached transform to get coords in neighbour cell
                    T = transforms[parent_cell, local_facet, :, :gdim, 0]
                    b = transforms[parent_cell, local_facet, :, gdim, 0]

                    X_new = T @ ref_coords_register[global_i] + b

                    # Store updated coords
                    ref_coords_register[global_i] = X_new
                    """

                # Compute reference coordinates in the new parent cells
                # TODO: Remove this step by pre-computing the coordinate transforms for all pairs of cells
                # For each cell, precompute neighbouring cell store in an integer field of size num_facets
                # pre compute coordinate transforms (A,b) and store in a matrix field
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
            
            # breakpoint()
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

        # Now update the VOM by removing all boundary particles
        # i.e., particles that have hit an exterior boundary in one of the iterations above.
        # This causes the VOM topology to change.

        # Rebuild the VOM given the updated particle positions
        # NOTE: Physical coordinates can be obtained from reference coordinates by interpolating the parent mesh into the VOM
        # Interpolation makes use of the parent cell ownership information and reference coordinates of VOM points
        new_phys_coords = assemble(interpolate(SpatialCoordinate(mesh), pmesh.coordinates.function_space()))
        pmesh_updater.rebuild_vom(absorbed_vom_indices=boundary_particles, new_coords=new_phys_coords.dat.data_ro)

        t += dt

    return t

def advance_ref_coords_euler(ref_pos_fn, invJ_vom, v_fn, dt_fn):
    """
    Advance particles forward by one Euler step in reference space.

    X(t + dt) = X(t) + J^-1*v*dt

    To distinguish between active and inactive particles within the inner loop,
    pass dt_fn as a DG0 Function storing per-particle time steps.
    """
    # Mesh consistency checks
    m = ref_pos_fn.function_space().mesh()
    assert invJ_vom.function_space().mesh() == m
    assert v_fn.function_space().mesh() == m
    assert dt_fn.function_space().mesh() == m
    
    update_expr = ref_pos_fn + invJ_vom * v_fn * dt_fn
    new_ref_pos_fn = assemble(interpolate(update_expr, ref_pos_fn.function_space()))
    return new_ref_pos_fn

def bisect_crossing_time_simd(
        ref_coords_fn,
        invJ_vom,
        v_fn,
        dt_left,
        ref_cell, 
        failed_global,
        FS_vom,
        tol=1e-12,
        max_iters=30
):
    """SIMD-style bisection algorithm that detects particle crossings.
    
    Instead of reconstructing midpoints using a hard-coded linear interpolation,
    this function evaluates midpoint positions by re-running the integrator.

    Returns crossing times and reference coordinates at the crossing point for failed particles.
    """
    n_failed = len(failed_global)

    # Per particle bisection brakets [t_lo, t_hi]
    t_lo = np.zeros(n_failed, dtype=float)
    t_hi = dt_left[failed_global].copy()

    # Define a DG0 timestep Function
    dt_mid_fn = Function(FS_vom)

    for _ in range(max_iters):
        t_mid = (t_lo + t_hi) / 2
        dt_mid_fn.dat.data[failed_global] = t_mid

        # Advance only failed particles by mid time substep
        mid_ref_fn = advance_ref_coords_euler(ref_coords_fn, invJ_vom, v_fn, dt_mid_fn)
        X_mid = mid_ref_fn.dat.data[failed_global]
        bary_mid = ref_cell.compute_barycentric_coordinates(X_mid)
        inside = np.all(bary_mid >= -tol, axis = 1)
        
        # For particles inside at the midpoint, advance lower end of the bracket
        t_lo[inside] = t_mid[inside]
        # For particels outside at the midpoing, advance higher end of the bracket
        t_hi[~inside] = t_mid[~inside]
        
        # Early exit if all brackets shrink sufficiently
        if np.max(t_hi - t_lo) < tol:
            break
    
    # Extract crossing times
    t_cross = t_lo

    # Compute barycentric coordinates at the crossing point
    dt_cross_fn = Function(FS_vom)
    dt_cross_fn.dat.data[failed_global] = t_cross
    cross_ref_fn = advance_ref_coords_euler(ref_coords_fn, invJ_vom, v_fn, dt_cross_fn)
    X_cross = cross_ref_fn.dat.data_ro[failed_global]
    bary_cross = ref_cell.compute_barycentric_coordinates(X_cross)

    return t_cross, bary_cross, X_cross

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
