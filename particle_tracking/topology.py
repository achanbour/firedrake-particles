def find_next_cell(mesh, cell, local_facet_id):
    """Find the neighbouring cell across a given facet of cell.
    
    Returns None if the facet is on the domain boundary.
    """
    facet_info = mesh.cell_to_facets.data_ro[cell, local_facet_id] # shape (num_cells, num_local_facets, 1)
    is_exterior = facet_info[0]
    if not bool(is_exterior):
        return None # boundary facets
    
    # `interior_facets.facet_cell` returns the two cells incident to each interior facet
    facet_cells = mesh.interior_facets.facet_cell # shape (num_interior_facets, 2) -> two cells incident to each interior facet.
    local_facet_numbers = mesh.interior_facets.local_facet_dat.data_ro # shape (num_interior_facets, 2) -> local facet ID in each adjacent cell.

    for f in range(facet_cells.shape[0]):
        c0, c1 = facet_cells[f]
        lf0, lf1 = local_facet_numbers[f]
        if c0 == cell and lf0 == local_facet_id:
            return c1
        if c1 == cell and lf1 == local_facet_id:
            return c0
    
    # Should never happen for a valid interior facet
    raise RuntimeError(
        f"Interior facet not found for cell {cell}, local facet {local_facet_id}"
    )

