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

        # Get the current parent cell ownership
        parent_cells = pmesh.topology.cell_parent_cell_list # ID of containing cell for each point in VOM order

        # Compute domain-safe time step dt
        # new_dt = compute_domain_safe_dt(pmesh.coordinates.dat.data_ro, v_fn, dt)
        # if new_dt != dt:
        #     print(f"Adjusted time step from original dt:{dt} to new dt:{new_dt} to ensure particles remain in domain.")
        #     dt = new_dt 

        # Per-particle tracking loop variables
        dt_left = np.full(N, dt) # remaining time for the current time step

        # Run outer loop while there are active particles (those that have not yet finished their dt)
        outer_loop_iter = 0
        breakpoint()
        while True:
            ref_coords_register = ref_coords_fn.dat.data_ro.copy() # copy the current reference coords for book-keeping throughout loops

            active = dt_left > 0
            if not np.any(active):
                break

            outer_loop_iter += 1
            active_indices = np.where(active)[0]

            # -- Phase 0 --
            # For all currently active particles, compute updated positions and detect crossings
            dt_trial_fn = Function(FS_vom)
            dt_trial_fn.dat.data[active_indices] = dt_left[active_indices]
            invJ_vom.interpolate(invJ_expr) # (re)compute invJ on the CURRENT embedding
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

            # Indices of particles that passed and failed
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

            print("\n=== POST PHASE 0 ===")
            print("Outer loop iteration:", outer_loop_iter)
            print(f"\nActive particles: {active_indices}")
            print(f"  Failed set: {failed_global}")
            print(f"  Passed set: {passed_global}")

            # -- Process passed particles --
            if len(passed_global) > 0:
                dt_left[passed_global] = 0
                ref_coords_register[passed_global] = trial_ref_pos_fn.dat.data_ro[passed_global]

                print("\nPassed set info:")
                print(f"  dt_left: {dt_left[passed_global]}")
                print(f"  new ref_coords: {ref_coords_register[passed_global]}")

            # -- Process failed particles -- 
            # Provided that `detect_crossings` correctly identifies the first crossing event,
            # we can now directly move failed particles to their crossing facets.
            if len(failed_global) > 0:
                dt_left[failed_global] -= t_cross[failed_global]
                # Get particle positions at crossing facet
                new_ref_coords_failed = move_failed_particles_to_facet(
                    failed_global, t_cross[failed_global], ref_coords_fn, invJ_vom, v_fn, FS_vom
                )
                ref_coords_register[failed_global] = new_ref_coords_failed

                print("Failed set info:")
                print(f"  dt_left: {dt_left[failed_global]}")
                print(f"  new ref_coords: {ref_coords_register[failed_global]}")

            else:
                print("Skipping phase 1 as there are no failed particles.")

            # At this point all the particles are now marked as passed.
            # 1) Run barycentric test again to ensure all particles are located within their original parent cells at their new positions.
            bary_register = ref_cell.compute_barycentric_coordinates(np.array(ref_coords_register))
            tol = 1e-12
            for i, coords_i in enumerate(bary_register):
                if np.any(coords_i < -tol):
                    print(f"Error: Particle {i} is outside its original parent cell at its new location.")
                    break
            else:
                print("\nPassed barycentric test: all particles are within their original parent cells at their new locations.")
            
            # 2) Move particles to neighbouring cells using the crossed_edges info
            # - Find neighbouring cells
            next_parent_cells = np.full(N, -1, dtype=int)
            # FIXME: `next_parent_cells` is initialized as a 1D array of shape (num_vertices,) where Firedrake's internal
            # `cell_parent_cell_list` is a 2D array of shape (num_vertices, 1). It may be desirable to keep the same tensor format for consistency.

            for i, parent_cell in enumerate(parent_cells):
                parent_cell = parent_cell[0] # extract cell ID from array
                edge_id = crossed_edges[i]
                if edge_id is None:
                    # The particle did not cross any edge so it stays in the same cell
                    next_parent_cells[i] = parent_cell
                    continue
                
                # Find neighbouring cell across the crossed edge
                next_cell = find_next_cell(mesh, parent_cell, edge_id)
                if next_cell is None:
                    print(f"Particle {i} attempted to cross boundary facet from cell {parent_cell}")
                    # keep next_parent_cells[i] as -1 to indicate boundary hit
                else:
                    next_parent_cells[i] = next_cell

            # - modify parent cell ownership in VOM
            # - update the reference coordinates Function (otherwise the next assemble/interpolate update will give wrong results)
            pmesh_updater.update_ref_view(next_parent_cells, ref_coords_register)

            # - recompute inverse Jacobian using new parent cell ownerhsip (done at start of outer loop)

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
    """Detect first crossings based on barycentric coordinates."""

    n_particles, n_verts = bary_new.shape
    passed = np.ones(n_particles, dtype=bool)
    t_cross = np.full(n_particles, np.inf)
    crossed_edges = np.full(n_particles, None, dtype=object)

    for i in range(n_particles):
        dt = dt_left[i]
        candidates = []

        for j in range(n_verts):
            lambda_old = bary_old[i, j]
            lambda_new = bary_new[i, j]

            # For a crossing to occur, a particle must start strictly inside and end strictly outside
            # if we allow lambda_tol == 0 then particles starting on a facet are misdetected as crossing
            if lambda_old > tol and lambda_new < -tol:
                if lambda_old - lambda_new <= tol:
                    continue
                t_j = dt * lambda_old / (lambda_old - lambda_new)

                if tol < t_j <= dt + tol:
                    candidates.append((t_j, j))
        
        if not candidates:
            continue # no crossing for this particle
            
        # Detect first crossing event
        passed[i] = False
        t_hit, neg_vert = min(candidates, key=lambda x: x[0])
        t_cross[i] = t_hit

        # Find edge opposite the vertex neg_vert
        for edge_id, edge_verts in edges.items():
            if neg_vert not in edge_verts:
                crossed_edges[i] = edge_id
                break

    return passed, t_cross, crossed_edges

def move_failed_particles_to_facet( failed_global, t_cross, ref_coords_fn, invJ_vom,v_fn, FS_vom):
    """Move failed particles exactly to their first crossing facet."""

    dt_step_fn = Function(FS_vom)
    dt_step_fn.dat.data[:] = 0.0
    dt_step_fn.dat.data[failed_global] = t_cross # set dt to t_cross for failed particles only

    # One single update
    ref_step_fn = update_ref_pos(ref_coords_fn, invJ_vom, v_fn, dt_step_fn)

    # Extract results for failed particles only
    ref_coords_final = ref_step_fn.dat.data_ro[failed_global].copy()

    return ref_coords_final


def compute_domain_safe_dt(coords, v_fn, dt, *, xmin=0.0, xmax=1.0, 
                           ymin=0.0, ymax=1.0, eps=1e-14):
    """Compute a domain-safe time step to ensure particles remain in the domain.
    NOTE: This has the drawback of being over restrictive since one bad particle can lead to a very small dt
    and the same dt is applied to all particles.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    v = v_fn.dat.data_ro

    d = np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y]) # smallest distance to domain boundary
    v_inf = np.maximum(np.abs(v[:, 0]), np.abs(v[:, 1])) # infinity norm of velocity
    dt_i = d / (v_inf + eps)
    return min(dt, np.min(dt_i))

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

    # Update regime 1: Move particles in ref. space
    T_final = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0)
    print("Final particle positions in physical space updated in ref space (in updated VOM order): ", particle_vom.coordinates.dat.data)
