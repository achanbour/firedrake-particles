import numpy as np

def find_next_cell(mesh, 
                   cell, 
                   edge_id):
    """
    Find the neighbouring cell across a given facet of a cell.
    
    Returns None if the facet is on the domain boundary.
    """
    # NOTE: 
    # - DMPlex gives rank local numbers of the edges (cones of the cell we're currently in).
    # - Look up the next cell using the support of the edge crossed -> PETSc local ID of the next cell.
    # - Look up Firedrake's cell number in the DMPlex. 

    plex = mesh.topology_dm

    # Get the DMPlex cell number for the current Firedrake cell
    # `cell_closure` provides the mapping from Firedrake entity indices -> DMPlex point numbering
    # `cell_closure` lists entities in increasing dimension order, first vertices then edges and finally facets/cells
    plex_cell = mesh.topology.cell_closure[cell, -1]

    # Compute the offset to get the edge point number in DMPlex
    num_vertices = mesh.ufl_cell().num_vertices
    plex_edge = mesh.topology.cell_closure[cell, num_vertices + edge_id]

    # Find the neighbouring cell using the support of this edge
    # (support gives entities one dimension higher)
    support_size = plex.getSupportSize(plex_edge)

    if support_size < 2:
        return None # boundary facet
    
    # Get the cells sharing this edge
    support = plex.getSupport(plex_edge)

    # Find the neighbour (cell that isn't the current one)
    plex_cell_numbers = mesh.topology._cell_numbering
    for plex_neigh in support:
        if plex_neigh != plex_cell:
            # Convert back to Firedrake cell numbering
            # `._cell_numbering` provides the mapping DMPlex point numbering -> Firedreake entity indices
            neighbour_cell = plex_cell_numbers.getOffset(plex_neigh)
            return neighbour_cell


def compute_ref_coords_in_new_cell(failed_global,
                                   parent_cells,
                                   new_parent_cells,
                                   crossed_edges,
                                   ref_coords,
                                   mesh,
                                   ref_cell):
    """
    Compute reference coordinates of particles in their new parent cells.

    Args:
        failed_global: Global indices of particles that crossed facets
        parent_cells: Current parent cell IDs 
        new_parent_cells: New parent cell IDs after crossing
        crossed_edges: Local facet IDs crossed in the current cells
        ref_coords: Reference coordinates in the current cell (at the crossing facet)
        mesh: The parent mesh
        ref_cell: FIAT reference cell
    
    Returns:
        Array of reference coordinates in the new cells
    """
    gdim = mesh.geometric_dimension
    facet_dim = ref_cell.get_spatial_dimension() - 1

    new_ref_coords = np.zeros((len(failed_global), gdim))

    plex = mesh.topology_dm
    num_vertices_per_cell = mesh.ufl_cell().num_vertices

    for l_pid, g_pid in enumerate(failed_global):
        current_cell = parent_cells[g_pid, 0]
        new_cell = new_parent_cells[g_pid, 0]
        crossed_edge_id = crossed_edges[l_pid] 
        current_coord = ref_coords[g_pid]

        # Get DMPlex point numbers
        plex_new_cell = mesh.topology.cell_closure[new_cell, -1]
        plex_crossed_edge = mesh.topology.cell_closure[current_cell, num_vertices_per_cell + crossed_edge_id]

        # Find the local ID of the crossed edge in the new cell using the cone of the new cell
        # reference cell entity numbering 
        new_cell_cone = plex.getCone(plex_new_cell)
        new_crossed_edge_id = None
        for l_eid, cone_point in enumerate(new_cell_cone):
            if cone_point == plex_crossed_edge:
                new_crossed_edge_id = l_eid 

        # Invert the entity transform to get facet-local coordinates from the current cell coordinates
        current_transform = ref_cell.get_entity_transform(facet_dim, crossed_edge_id)
        current_coord_on_facet = get_facet_coord(
            current_transform, 
            current_coord, 
            facet_dim,
            ref_cell
        )

        # Get the entity transform in the new cell allowing us to map facet-local coordinates to the new cell coordinates
        # x_new_cell = A_new * x_facet + b_new
        new_transform = ref_cell.get_entity_transform(facet_dim, new_crossed_edge_id)
        new_ref_coords[l_pid] = new_transform(current_coord_on_facet)

    return new_ref_coords

def get_facet_coord(entity_transform, ref_coord, facet_dim, ref_cell):
    """
    Invert the entity transform to get facet-local coordinates.
    
    Given the entity transform affine map x_cell = A * x_facet + b, solve for x_facet.
    """
    # Apply the entity transform at facet vertices to recover A and b
    facet_ref_element = ref_cell.construct_subelement(facet_dim)
    facet_verts = np.array(facet_ref_element.get_vertices())
    transformed = np.array([entity_transform(v) for v in facet_verts])
    b = transformed[0] 
    A = (transformed[1:] - b).T

    # NOTE: A is non-square (for 2D mesh, maps 1D edge coords to 2D cell coords)
    # but since the particle is exactly on the facet, the project is exact
    x_facet = np.linalg.lstsq(A, ref_coord - b, rcond=None)[0]
    
    return x_facet

# def compute_ref_coords_in_new_cell(failed_global,
#                                    parent_cells,
#                                    next_parent_cells,
#                                    crossed_edges_local,
#                                    bary_coords,
#                                    mesh,
#                                    ref_cell
#                                    ):
#     """
#     Compute the reference coordinates of a particle in its new containing cell.
    
#     When a particle crosses a facet from one cell to another, we recompute its barycentric
#     coordinates in the current cell (expressed in terms of the current cell's vertices).
#     To get its reference coordinates in the new cell, we:
#     1. Find the shared vertices on the crossed facet
#     2. Map local vertex indices from the current cell to the new cell using global vertex IDs
#     3. Construct the barycentric coordinates vector in the new cell
#     4. Convert barycentric coordinates to reference coordinates
#     """

#     # Get reference cell topology and vertex coordinates
#     ref_cell_edges = ref_cell.get_topology()[1]  
#     ref_cell_vertices = ref_cell.vertices
#     n_vertices = len(ref_cell_vertices)
#     gdim = mesh.geometric_dimension
    
#     ref_coords_in_new_cell = np.zeros((len(failed_global), gdim))
    
#     for idx, global_particle_id in enumerate(failed_global):
#         current_cell = parent_cells[global_particle_id]
#         next_cell = next_parent_cells[global_particle_id]
#         crossed_edge_id = crossed_edges_local[idx]
        
#         # Local vertex IDs forming the crossed edge in the ref cell
#         local_vids_in_crossed_edge = ref_cell_edges[crossed_edge_id]
        
#         # Get the global vertex IDs in the current cell
#         global_vids_current = mesh.coordinates.function_space().cell_node_list[current_cell].ravel() 
#         global_crossed_edge_verts = [global_vids_current[v] for v in local_vids_in_crossed_edge]
        
#         # Get the global vertex IDs in the next cell
#         global_vids_next = mesh.coordinates.function_space().cell_node_list[next_cell].ravel()
        
#         # Build barycentric coordinates in the new cell
#         # For each vertex on the shared edge, map its barycentric coordinate
#         # from its position in the current cell to its position in the new cell
#         bary_coords_new = np.zeros(n_vertices)
        
#         for local_vid_current, global_v_current in zip(local_vids_in_crossed_edge, global_crossed_edge_verts):
#             # Find which local vertex index in the new cell corresponds to the shared global vertex ID
#             for local_vid_next, global_v_next in enumerate(global_vids_next):
#                 if global_v_next == global_v_current:
#                     # Map the barycentric coordinate
#                     bary_coords_new[local_vid_next] = bary_coords[global_particle_id, local_vid_current]
#                     break
        
#         # Convert barycentric coordinates to reference coordinates
#         ref_coords_in_new_cell[idx] = np.dot(bary_coords_new, ref_cell_vertices)
    
#     return ref_coords_in_new_cell

# def find_next_cell(mesh, 
#                    cell, 
#                    local_facet_id):
#     """
#     Find the neighbouring cell across a given facet of a cell.
    
#     Returns None if the facet is on the domain boundary.
#     """
#     # NOTE: 
#     # - DMPlex gives rank local numbers of the edges (cones of the cell we're currently in).
#     # - Look up the next cell using the support of the edge crossed -> PETSc local ID of the next cell.
#     # - Look up Firedrake's cell number in the DMPlex. 

#     # `cell_to_facets` is a PyOp2 Dat that maps each cell to its (local) facets.
#     # The i-th local facet of cell c has data stored in cell_to_facets[c][i]
#     # which returns a list [is_exterior, subdomain marker]
#     cell_facet_data = mesh.cell_to_facets.data_ro[cell][local_facet_id]
#     is_exterior = cell_facet_data[0]

#     if not bool(is_exterior):
#         return None # boundary facet
    
#     # `interior_facets.facet_cell` returns the two cells incident to each interior facet
#     interior_cells = mesh.interior_facets.facet_cell # (num_interior_facets, 2) -> returns the two cells incident to each interior facet.
#     local_facet_numbers = mesh.interior_facets.local_facet_dat.data_ro # (num_interior_facets, 2) -> returns the local facet ID in each adjacent cell.

#     for f in range(interior_cells.shape[0]):
#         c0, c1 = interior_cells[f] # get the two cells adjacent to facet f
#         lf0, lf1 = local_facet_numbers[f] # get the local facet ID in each adjacent cell

#         # The neighbour cell is the one that is not equal to `cell` by exclusion
#         if c0 == cell and lf0 == local_facet_id:
#             return c1
#         if c1 == cell and lf1 == local_facet_id:
#             return c0
    
#     # This should never happen for a valid interior facet
#     raise RuntimeError(
#         f"Interior facet not found for cell {cell}, local facet {local_facet_id}"
#     )