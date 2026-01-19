import numpy as np

def find_next_cell(mesh, 
                   cell, 
                   local_facet_id):
    """Find the neighbouring cell across a given facet of a cell.
    
    Returns None if the facet is on the domain boundary.
    """

    # `cell_to_facets` is a PyOp2 Dat that maps each cell to its (local) facets.
    # The i-th local facet of cell c has data stored in cell_to_facets[c][i]
    # which returns a list [is_exterior, subdomain marker]
    cell_facet_data = mesh.cell_to_facets.data_ro[cell][local_facet_id]
    is_exterior = cell_facet_data[0]
    if not bool(is_exterior):
        return None # boundary facet
    
    # `interior_facets.facet_cell` returns the two cells incident to each interior facet
    interior_cells = mesh.interior_facets.facet_cell # (num_interior_facets, 2) -> returns the two cells incident to each interior facet.
    local_facet_numbers = mesh.interior_facets.local_facet_dat.data_ro # (num_interior_facets, 2) -> returns the local facet ID in each adjacent cell.

    for f in range(interior_cells.shape[0]):
        c0, c1 = interior_cells[f] # get the two cells adjacent to facet f
        lf0, lf1 = local_facet_numbers[f] # get the local facet ID in each adjacent cell

        # The neighbour cell is the one that is not equal to `cell` by exclusion
        if c0 == cell and lf0 == local_facet_id:
            return c1
        if c1 == cell and lf1 == local_facet_id:
            return c0
    
    # This should never happen for a valid interior facet
    raise RuntimeError(
        f"Interior facet not found for cell {cell}, local facet {local_facet_id}"
    )


def compute_ref_coords_in_new_cell(failed_global,
                                   parent_cells,
                                   next_parent_cells,
                                   crossed_edges_local,
                                   bary_coords,
                                   mesh,
                                   ref_cell
                                   ):
    """
    Compute the reference coordinates of particles in their new cells.
    
    When a particle crosses a facet from one cell to another, we have its barycentric
    coordinates in the current cell (expressed in terms of the current cell's vertices).
    To get its reference coordinates in the new cell, we:
    1. Find the shared vertices on the crossed facet
    2. Map local vertex indices from the current cell to the new cell
    3. Construct the barycentric coordinates vector in the new cell
    4. Convert barycentric coordinates to reference coordinates
    """
    # Get reference cell topology and vertex coordinates
    ref_cell_edges = ref_cell.get_topology()[1]  
    ref_cell_vertices = ref_cell.vertices
    n_vertices = len(ref_cell_vertices)
    gdim = mesh.geometric_dimension
    
    ref_coords_in_new_cell = np.zeros((len(failed_global), gdim))
    
    for idx, global_particle_id in enumerate(failed_global):
        current_cell = parent_cells[global_particle_id]
        next_cell = next_parent_cells[global_particle_id]
        crossed_edge_id = crossed_edges_local[idx]
        
        # Local vertex IDs forming the crossed edge in the ref cell
        local_vids_in_crossed_edge = ref_cell_edges[crossed_edge_id]
        
        # Get the global vertex IDs in the current cell
        global_vids_current = mesh.coordinates.function_space().cell_node_list[current_cell].ravel() 
        global_edge_verts_current = [global_vids_current[v] for v in local_vids_in_crossed_edge]
        
        # Get the global vertex IDs in the next cell
        global_vids_next = mesh.coordinates.function_space().cell_node_list[next_cell].ravel()
        
        # Build barycentric coordinates in the new cell
        # For each vertex on the shared edge, map its barycentric coordinate
        # from its position in the current cell to its position in the new cell
        bary_new_cell = np.zeros(n_vertices)
        
        for local_vid_current, global_v_current in zip(local_vids_in_crossed_edge, global_edge_verts_current):
            # Find which local vertex index in the new cell corresponds to the shared global vertex ID
            for local_vid_next, global_v_next in enumerate(global_vids_next):
                if global_v_next == global_v_current:
                    # Map the barycentric coordinate
                    bary_new_cell[local_vid_next] = bary_coords[global_particle_id, local_vid_current]
                    break
        
        # Convert barycentric coordinates to reference coordinates
        ref_coords_in_new_cell[idx] = np.dot(bary_new_cell, ref_cell_vertices)
    
    return ref_coords_in_new_cell
