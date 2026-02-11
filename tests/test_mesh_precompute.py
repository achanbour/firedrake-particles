from firedrake import *
import pytest
import numpy as np

# `mesh.interior_facets` is a _Facets object
# `mesh.interior_facets.facet_cell` gives the 2 cells adjacent to each facet # (num_facets, 2)
# `mesh.interior_facets.local_facet_dat` gives the local facet ID of a facet in each of its 2 adjacent cells #(num_facets, 2)

@pytest.mark.parametrize("quadrilateral, num_facets", [
    (False, 3), # triangles
    (True, 4) # quads
])
def test_precomputed_array_sizes(quadrilateral, num_facets):
    mesh = UnitSquareMesh(10, 10, quadrilateral=quadrilateral)
    cell_neighbours, local_cell_coords_transforms = mesh.precompute_cell_neighbours_and_transforms()

    assert cell_neighbours.shape == (mesh.num_cells(), num_facets)
    assert local_cell_coords_transforms.shape == (mesh.num_cells(), num_facets, mesh.geometric_dimension, mesh.geometric_dimension + 1, 2)


@pytest.mark.parametrize("quadrilateral, num_facets", [
    (False, 3), # triangles 
    (True, 4) # quads 
])
def test_precomputed_coordinate_transforms(quadrilateral, num_facets, tol=1e-12):
    mesh = UnitSquareMesh(10, 10, quadrilateral=quadrilateral)
    gdim = mesh.geometric_dimension
    cell_neighbours, local_cell_coords_transforms = mesh.precompute_cell_neighbours_and_transforms()
    
    # Find a cell which has a neighbour across an interior facet
    found = False   
    for c in range(mesh.num_cells()):
        for lf_c in range(num_facets):
            n = cell_neighbours[c, lf_c]
            if n != -1:
                found = True
                break
        if found:
            break
    
    # Get local facet ID in the neighbouring cell
    lf_n = np.where(cell_neighbours[n] == c)[0][0]

    # Define a point on on the local facet and map that point to the cell   
    import FIAT
    ref_cell = FIAT.ufc_cell(mesh.ufl_cell())
    phi = ref_cell.get_entity_transform(gdim - 1, lf_c)

    midpoint = np.asarray([0.5]) # midpoint
    x_c = phi(midpoint) 

    # Compute coords. of the point in the neighbouring cell under the forward transform (c -> n)
    A = local_cell_coords_transforms[c, lf_c, :, :gdim, 0]
    b = local_cell_coords_transforms[c, lf_c, :, gdim, 0]
    x_n = A @ x_c + b

    # Check if x_n is in cell n and vanishes on local facet lf_n
    bary_coords = ref_cell.compute_barycentric_coordinates(x_n)
    assert np.all(bary_coords >= - tol)
    assert abs(bary_coords[0, lf_n]) <= tol

    # Check backward transform (n -> c)
    A_back = local_cell_coords_transforms[c, lf_c, :, :gdim, 1]
    b_back = local_cell_coords_transforms[c, lf_c, :, gdim, 1]

    x_back = A_back @ x_n + b_back
    assert np.allclose(x_back, x_c, atol=tol)









    

