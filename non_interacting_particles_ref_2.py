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

        # Run outer loop while there are active particles (those that have not yet finished their dt)
        outer_loop_iter = 0
        active_iters = np.zeros(N, dtype=int)

        while True:
            # Ensure ref_coords_register is equal to the "latest" ref coords
            if not np.array_equal(ref_coords_register, ref_coords_fn.dat.data_ro):
                ref_coords_register = ref_coords_fn.dat.data_ro.copy()

            # Check if there any active particles left
            active = dt_left > 0
            if not np.any(active):
                break

            outer_loop_iter += 1
            active_indices = np.where(active)[0]
            active_iters[active_indices] += 1

            # -- Process active particles --
            # For all currently active particles, compute updated positions and detect crossings
            dt_trial_fn = Function(FS_vom)

            # NOTE: this line enforces that dt_left indexes particles in VOM ordering 
            # i.e., dt_left[i] corresponds to VOM particle i
            dt_trial_fn.dat.data[active_indices] = dt_left[active_indices]
            invJ_vom.interpolate(invJ_expr) # recompute invJ on the CURRENT embedding
            trial_ref_pos_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_trial_fn)

            # Compute barycentric coordinates based on ref cell type
            if isinstance(ref_cell, (UFCInterval, UFCTriangle, UFCTetrahedron)):
                # For simplex-based cells, compute vertex barycentric coordinates directly
                bary_old = ref_cell.compute_barycentric_coordinates(ref_coords_fn.dat.data_ro)
                bary_new = ref_cell.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro)

            elif isinstance(ref_cell, (UFCQuadrilateral, UFCHexahedron)):
                # For tensor-product cells, compute barycentric coordinates per axis
                tp_cell = ref_cell.product # unflattened tensor product element
                axes = tp_cell.cells
                slices = tp_cell._split_slices([c.get_dimension() for c in axes])

                bary_old_per_axis = []
                bary_new_per_axis = []
                for axis, slice in zip(axes, slices):
                    bary_old_per_axis.append(
                        axis.compute_barycentric_coordinates(ref_coords_fn.dat.data_ro[:, slice])
                    )
                    bary_new_per_axis.append(
                        axis.compute_barycentric_coordinates(trial_ref_pos_fn.dat.data_ro[:, slice])
                    )
                # Stack the axes barycentric coordinates
                bary_old = np.hstack(bary_old_per_axis)
                bary_new = np.hstack(bary_new_per_axis)

            else:
                raise NotImplementedError(
                    f"Barycentric coordinate computation not implemented for cell type {type(ref_cell)}"
                )
            
            # Detect crossings and split particles into passed/failed subsets
            passed_mask, t_cross_local, crossed_edges_local = detect_crossings_linear(
                bary_old[active_indices], bary_new[active_indices], dt_left[active_indices], ref_cell
            )

            # Get local indices (in the currently active set) of passed/failed particles
            passed_local = np.where(passed_mask)[0]
            failed_local = np.where(~passed_mask)[0]

            # Map to global indices (in the full particle array)
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
                - update parent cell to neighbour across the crossed facet
                - re-enter outer loop as active with updated ref. pos., parent cell and dt_left
            """
            print(f"\n---Outer loop iteration: {outer_loop_iter}---")
            print(f"Active particles: {active_indices}")
            print(f"  Failed set: {failed_global}")
            print(f"  Passed set: {passed_global}")

            # -- Process passed and failed particles separately --
            # Passed particles
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            # Failed particles
            if len(failed_global) > 0:
                print("\nMoving failed particles to crossing facets...")
                # Compute particle positions at the crossing facet
                new_ref_coords_at_facet = move_failed_particles_to_facet(
                    failed_global, t_cross_local[failed_local], ref_coords_fn, invJ_vom, v_fn, FS_vom
                )
                dt_left[failed_global] -= t_cross_local[failed_local]
                ref_coords_register[failed_global] = new_ref_coords_at_facet

                print("Failed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref coords (in current cell): {ref_coords_register[failed_global]}")

                # Validate the new particle positions:
                # - Passed particles should be inside their original parent cells
                # - Failed particles should be on a facet of their original parent cells (one barycentric coord. is zero)

                # print("\nComputing barycentric coordinates at the new reference positions...")
                # new_bary_coords = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_register))
                # tol = 1e-12
                # for global_i in passed_global:
                #     if np.any(new_bary_coords[global_i] < -tol):
                #         print(f"Error: Passed particle {global_i} is outside its cell.")
                
                # for global_i in failed_global:
                #     if not np.any(np.abs(new_bary_coords[global_i]) < tol):
                #         print(f"Warning: Failed particle {global_i} is not on a facet.")
                #     if np.any(new_bary_coords[global_i] < -tol):
                #         print(f"Error: Failed particle {global_i} has been moved past the facet.")
                # print("Barycentric validation complete.")

                # NOTE: Skipping the above as it currently works only for simplex cells.

                breakpoint()
                # 2) Move failed particles to neighbouring cells using the crossed_edges_local info
                print("\nSearching for next parent cells for failed particles...")
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
                print("All neighbouring cells determined.")

                # 3) Compute reference coordinates in the new parent cells
                # For failed particles at the crossing facets, map their barycentric coordinates
                # from the current cells to the neighboring cells, then convert to reference coordinates

                breakpoint()
                print("\nComputing new reference coordinates in the neighbouring cells...")
                new_ref_coords_in_new_cells = compute_ref_coords_in_new_cell(
                    failed_global,
                    parent_cells,
                    next_parent_cells,
                    crossed_edges_local[failed_local],
                    new_bary_coords,
                    mesh,
                    ref_cell
                )
                ref_coords_register[failed_global] = new_ref_coords_in_new_cells
                print(f"  new ref coords (in next cells): {ref_coords_register[failed_global]}")

            # 4) Update the particle VOM:
            # - modify parent cell ownership
            # - update the reference coordinates (otherwise the next assemble/interpolate update will give wrong results)
            pmesh_updater.update_ref_view(next_parent_cells, ref_coords_register)

            # - recompute inverse Jacobian using new parent cell ownership (done at start of outer loop)
            # 5) Re-enter the outer loop with new ref. coords., parent cells and remaining dt_left

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
        # Now update the VOM by removing all boundary particles
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

def detect_crossings_linear(bary_old, bary_new, dt_left, ref_cell, tol=1e-12):
    """
    An interval-based crossing detection algorithm that finds the last facet crossed by a particle in a given time step.

    This method is used instead of the root-finding approach which consists in finding the roots of lambda_j(t) for each 
    lambda_j then taking the max. among all crossing times.
    This approach is not correct since not all crossings are feasible (some may correspond to intersections outside the cell). 

    For each particle, compute the time interval during which the particle remained inside the cell 
    i.e., during which all barycentric coordinates are non negative. 
    The exit time is taken as the upper bound of this interval (giving us the last crossing).

    The arguments `bary_old`, `bary_new` and `dt_left` are assumed to refer to the currently active set of particles.
    The argument `ref_cell_edges` refer to the reference cell topology map which maps each local edge ID to the local vertex IDs forming that edge.
    """

    N_active, n_coords = bary_new.shape
    passed = np.ones(N_active, dtype=bool)
    t_cross = np.full(N_active, np.inf)
    crossed_edges = np.full(N_active, None, dtype=object)

    for i in range(N_active):
        dt = dt_left[i]
        lambda_old = bary_old[i]
        lambda_new = bary_new[i]
        dlambda = (lambda_new - lambda_old) / dt # derivative of the barycentric trajectory

        t_out = dt
        exit_coord = None
        feasible = True

        for j in range(n_coords):
            if abs(dlambda[j]) < tol:
                # lambda_j constant
                if lambda_old[j] < -tol:
                    feasible = False # particle starts outside the cell and remains outside
                    break
                continue

            t_zero = -lambda_old[j] / dlambda[j] # time when lambda_j becomes zero

            if dlambda[j] < 0:
                # lambda_j decreasing: particle exits the cell at t_zero
                if t_zero < t_out:
                    t_out = t_zero
                    exit_coord = j

        # No valid intersection with the cell
        # NOTE: a slightly negative t_out due to numerical roundoff can cause t_in > t_out so we allow a small tolerance here.
        if not feasible or t_out < -tol or t_out > dt + tol:
            passed[i] = False
            t_cross[i] = 0.0
            crossed_edges[i] = None
            continue
        
        # Particle remains inside cell for whole time step
        if t_out >= dt - tol:
            continue

        # Particle exits at t_out
        # NOTE: This includes the case where t_out is approx. 0 such as when a particle starts on a facet 
        # and moves outward through the same facet.
        passed[i] = False
        t_cross[i] = t_out

        # Identify the edge crossed based on the exit coordinate
        # For simplicies, this is the edge opposite the exit coordinate   
        # For tensor-product cells, we need a mapping that maps the exit coordinate to the corresponding facet
        coord_to_facet = {}
        if isinstance(ref_cell, (UFCInterval, UFCTriangle, UFCTetrahedron)):
            for edge_id, vertex_ids in ref_cell.get_topology()[1].items():
                for v in range(n_coords):
                    if v not in vertex_ids:
                        coord_to_facet[v] = edge_id
                        break

        elif isinstance(ref_cell, (UFCQuadrilateral, UFCHexahedron)):
            coord_to_facet = build_coord_to_facet_map_for_tensor_ref_cell(ref_cell, tol=tol)

        crossed_edges[i] = coord_to_facet[exit_coord]

    return passed, t_cross, crossed_edges
        

def move_failed_particles_to_facet( failed_global, t_cross, ref_coords_fn, invJ_vom,v_fn, FS_vom):
    """Move failed particles to their crossing facet."""

    dt_step_fn = Function(FS_vom)
    dt_step_fn.dat.data[failed_global] = t_cross # set dt to t_cross for failed particles only

    # Do one single update to bring particles to their crossing facet
    ref_step_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_step_fn)

    # Extract results for failed particles only
    ref_coords_at_facet = ref_step_fn.dat.data_ro[failed_global].copy()

    return ref_coords_at_facet

def build_coord_to_facet_map_for_tensor_ref_cell(ref_cell, tol=1e-12):
    """
    Return a mapping from the index in the stacked barycentric coords vector
    to the local facet ID of a tensor-product reference cell (e.g., UFCQuadrilateral).
    """
    coord_to_facet = {}

    facet_dim = ref_cell.get_spatial_dimension() - 1
    tp_cell = ref_cell.product
    axes = tp_cell.cells

    # Track where each axis's barycentric coordinates start in the stacked vector
    axes_offsets = []
    offset = 0
    for a in axes:
        axes_offsets.append(offset)
        offset += a.get_dimension() + 1

    # Slices for extracting sub-coordinates
    slices = tp_cell._split_slices([ax.get_dimension() for ax in axes])

    for facet_id in range(len(ref_cell.get_topology()[facet_dim])):
        phi = ref_cell.get_entity_transform(facet_dim, facet_id) # get the mapping from the facet local coords to the full cell coordsbreakpoint()
        midpoint = np.full((1, facet_dim), 0.5) # facet midpoint in facet ref coords
        mapped = phi(midpoint)[0] # facet midpoint in full cell ref coords

        # For each axis, compute which barycentric coordinate vanishes at the mapped point
        for i, (axis, slice) in enumerate(zip(axes, slices)):
            axis_coords = mapped[slice] # extract ref coordinates along this axis
            lambdas = axis.compute_barycentric_coordinates(axis_coords.reshape(1, -1))[0]

            for local_idx, val in enumerate(lambdas):
                if abs(val) < tol:
                    global_index = axes_offsets[i] + local_idx
                    coord_to_facet[global_index] = facet_id
                    break
                
    return coord_to_facet


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
    print("Final particle positions in physical space updated in ref space (in updated VOM order): ", particle_vom.coordinates.dat.data)
