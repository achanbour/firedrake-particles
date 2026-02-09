from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater
from moving_particles_phys_space import move_particles_in_phys_space
from ufl.differentiation import ReferenceGrad

np.random.seed(42)

t = 0.0
dt = 0.1
T = 1

def move_particles_in_ref_space(pmesh, mesh, v, dt, T, t=0.0):
    """
    Update particle positions in reference space using Forward Euler:
    
    X(t + dt) = X(t) + J^-1*v*dt

    where J = dF/dX is the Jacobian of the map from the ref. to physical space F: X -> x.
    """
    x = SpatialCoordinate(mesh)

    # invJ_expr = inv(grad(x)) # UFL expression for the Jacobian inverse
    invJ_expr = inv(ReferenceGrad(x))

    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    while t < T:
        coords = pmesh.coordinates # a Firedrake Function storing the physical coords. in VOM order)
        # coords_io = pmesh.input_ordering.coordinates # a Firedrake Function storing the physical coords. in input ordering
        ref_coords = pmesh.reference_coordinates # a Firedrake Function storing the reference coords. (in VOM order)
        
        print(f"t={t}: ref coords: {ref_coords.dat.data_ro}")

        W_vom = TensorFunctionSpace(pmesh, "DG", 0)
        invJ_vom = Function(W_vom)
        invJ_vom.interpolate(invJ_expr) # gives J^-1(x) evaluated at the VOM points using the geometry of their parent cell
    
        update_expr = ref_coords + invJ_vom * v * dt # UFL expression for the coordinates update in ref. space 

        new_ref_coords = assemble(interpolate(update_expr, ref_coords.function_space())) # a Firedrake Function storing the updated particle reference coords.
        print("new_ref_coords:", new_ref_coords.dat.data_ro)

        # -- Fill `cells_coords` array
        # which contains for each VOM point, the global coords. of the vertices of its containing cell
        cells = pmesh.topology.cell_parent_cell_list # ID of containing cell for each point in VOM order
        nverts = mesh.coordinates.cell_node_map().arity # num. cell vertices
        cells_coords = np.zeros((pmesh.num_vertices(), nverts, gdim)) # (num. points, num. cell vertices, gdim)
        for i, cell_num in enumerate(cells):
            cell_nodes = mesh.coordinates.cell_node_map().values[cell_num, :]
            cell_coords = mesh.coordinates.dat.data_ro[cell_nodes, :]
            cells_coords[i, :, :] = cell_coords

        # -- Update reference coordinates **manually**
        # To verify correctness of the UFL-based update above
        """
        new_ref_coords_computed = ref_coords.dat.data_ro.copy()
        for i, X_i in enumerate(ref_coords.dat.data_ro):
            v_i = v.dat.data_ro[i]
            x0, x1, x2 = cells_coords[i, :, :] # vertices of the parent mesh cells (phys. coords)
            J = np.column_stack((x1 - x0, x2 - x0)) # Jacobian of the map X -> x
            dX = np.linalg.solve(J, v_i) * dt # J^-1*v*dt
            new_ref_coords_computed[i] = X_i + dX
        print("new_ref_coords_computed:", new_ref_coords_computed)
        """
        
        # -- Evaluate basis functions at the new reference points
        # 1. to compute barycentric coordinates and check for each particle if it has left its containing cell
        # 2. to compute **new** physical coordinates

        # `new_ref_coords` has points ordered in VOM ordering
        # since it is a function of ref_coords.function_space()
        # which is a FS on `pmesh` (so inherits the VOM ordering)
        """
        basis_values = np.zeros((pmesh.num_vertices(), nverts)) # (num. points, num. cell vertices)       
        tol = 1e-12
        for i, ref_point in enumerate(new_ref_coords.dat.data_ro):
            w = np.array([
                1 - ref_point[0] - ref_point[1],
                ref_point[0],
                ref_point[1]
            ])
            if np.any(w < -tol):
                print(f"Particle {i} left its parent cell")
            basis_values[i, :] = w
        """

        ref_cell = mesh.coordinates.function_space().finat_element.cell
        topology = ref_cell.get_topology()
        edges = topology[1] # in a 2D cell, edges are dimension 1 entities
        old_bary_coords = ref_cell.compute_barycentric_coordinates(np.array(ref_coords.dat.data_ro))
        new_bary_coords = ref_cell.compute_barycentric_coordinates(np.array(new_ref_coords.dat.data_ro))

        dt_remaining = np.zeros(len(new_bary_coords)) # remaining time after crossing
        crossed_edges = [None] * len(new_bary_coords)  # local IDs of crossed edges
        
        tol = 1e-12
        for i, coords_i in enumerate(new_bary_coords):
            negative_verts = np.where(coords_i < -tol)[0]
            
            if len(negative_verts) > 0:
                for neg_vert in negative_verts:
                    # Find edge opposite this vertex
                    for edge_id, edge_verts in edges.items():
                        if neg_vert not in edge_verts:
                            # Compute intersection time
                            old_lambda = old_bary_coords[i, neg_vert]
                            new_lambda = coords_i[neg_vert]
                            t_cross = -old_lambda / (new_lambda - old_lambda) * dt
                            dt_remaining[i] = dt - t_cross
                            crossed_edges[i] = edge_id
                            break  
                    print(f"Particle {i} left its parent cell; edge={crossed_edges[i]}, remaining_time={dt_remaining[i]:.6f}")

        step_tuple = (new_ref_coords.dat.data_ro.copy(), dt_remaining, crossed_edges)
        print("Per-step tuple:", step_tuple)

        # TODO
        # For particles with remaining time dt_remaining:
        # 1. Update their parent mesh cell to the neighboring cell across the crossed edge
        # 2. Resume the update for the remaining -> get new ref coords.
        # 3. Do barycentric test again
        # 4. Repeat until all particles have used up their dt_remaining
        # Check if particle has hit the domain boundary every time? 
        # When going through the facet

        # -- Compute new physical coordinates
        new_phys_coords = np.einsum('in, ing->ig', new_bary_coords, cells_coords) # in VOM ordering
        # print("new_phys_coords: ", new_phys_coords)
        new_phys_coords_func = Function(coords.function_space())
        new_phys_coords_func.dat.data[:] = new_phys_coords

        # -- Update the VOM 
        # pass a Firedrake `Function` with updated physical coords. in VOM ordering
        # reordering happens internally within the updater
        pmesh_updater.update(new_phys_coords_func)

        t += dt

    return t

if __name__=='__main__':
    # Define the parent mesh
    mesh = UnitSquareMesh(10, 10)

    N = 10
    # particle_coords = np.random.rand(N, 2) 
    # Define a box within the mesh to place the particles in initially
    xmin, xmax = 0.2, 0.8
    ymin, ymax = 0.2, 0.8
    particle_coords = np.zeros((N, 2))
    particle_coords[:, 0] = xmin + (xmax - xmin) * np.random.rand(N)
    particle_coords[:, 1] = ymin + (ymax - ymin) * np.random.rand(N)
    print("Initial particle positions (in input order): ", particle_coords)

    # Define two particles VOM
    # One is to be modified by updating particle positions in reference space
    # The other is to be modified by updating particle positions in phyiscal space
    particle_vom = VertexOnlyMesh(mesh, particle_coords)
    particle_vom_copy = VertexOnlyMesh(mesh, particle_coords)
        
    # print("Particle VOM ID: ", particle_vom.ufl_id())
    # print("Particle VOM copy ID: ", particle_vom_copy.ufl_id())
    # particle_vom._dm_renumbering.view()

    gdim = particle_vom.geometric_dimension
    print("Initial particle positions (in primary VOM order): ", particle_vom.coordinates.dat.data_ro)

    assert np.all(particle_vom.coordinates.dat.data_ro == particle_vom_copy.coordinates.dat.data_ro)

    # Assign per-particle velocities in both VOMs
    V = VectorFunctionSpace(particle_vom, "DG", 0, dim=gdim)
    V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=gdim)

    V_copy = VectorFunctionSpace(particle_vom_copy, "DG", 0, dim=2)
    V_copy_io = VectorFunctionSpace(particle_vom_copy.input_ordering, "DG", 0, dim=gdim)

    v = Function(V)
    v_io = Function(V_io)

    v_copy = Function(V_copy)
    v_copy_io = Function(V_copy_io)

    input_velocities = np.random.normal(0.0, 0.5, size=(N,2))
    v_io.dat.data[:] = input_velocities
    v_copy_io.dat.data[:] = input_velocities

    v.interpolate(v_io)
    v_copy.interpolate(v_copy_io)

    # Set the parameters below to do a single integration step 
    T = 0.3
    dt = T

    # Update regime 1: Move particles in ref. space
    T_final = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0)
    print("Final particle positions in physical space updated in ref space (in updated VOM order): ", particle_vom.coordinates.dat.data)

    # Confirm that final coords match parent mesh interpolation onto the updated VOM
    # embedding_func = assemble(interpolate(x, particle_vom.coordinates.function_space()))
    # # print("Parent mesh embedding: ", embedding_func.dat.data)
    # print("Parent mesh embedding diff: ", embedding_func.dat.data - particle_vom.coordinates.dat.data)

    # Update regime 2: Move particles in physical space
    T_final = move_particles_in_phys_space(particle_vom_copy, mesh, v_copy, dt, T, t=0.0)
    print("Final particle positions in physical space (in updated VOM order): ", particle_vom_copy.coordinates.dat.data)

    # Check difference in coords. between the two update regimes
    print("Diff between updating in ref. space vs physical space", particle_vom.coordinates.dat.data - particle_vom_copy.coordinates.dat.data)
