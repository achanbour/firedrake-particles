import numpy as np
from FIAT.reference_element import UFCQuadrilateral, UFCHexahedron
        
def test_quad_smoke_test():
    ref = UFCQuadrilateral()

    x = np.array([[0.3, 0.7],[0.0, 0.7]]) # interior point + facet point
    bary_coords = ref.compute_barycentric_coordinates(x)

    # a quad has 4 facets
    assert bary_coords.shape == (2, 4)

    # all bary coords. are be non-neg
    assert np.all(bary_coords[0,:] > 0.0)
    assert np.all(bary_coords[1,:] >= 0.0)

def test_hex_smoke_test():
    ref = UFCHexahedron()
    x = np.array([[0.2, 0.4, 0.6]])
    bary_coords = ref.compute_barycentric_coordinates(x)
    
    # a hex has 6 facets
    assert bary_coords.shape == (1, 6)

    # all bary coords. are be non-neg
    assert np.all(bary_coords[0,:] > 0.0)

def test_quad_facet_vanishing():
    ref = UFCQuadrilateral()
    sd = ref.get_spatial_dimension()
    facet_dim = sd - 1

    for ufc_facet in ref.get_topology()[facet_dim]:
        phi = ref.get_entity_transform(facet_dim, ufc_facet) # map from facet coords. to cell coords
        x = phi(np.array([[0.5]]))  # midpoint of facet
        bary_coords = ref.compute_barycentric_coordinates(x)[0]

        assert abs(bary_coords[ufc_facet]) < 1e-12
        for j, val in enumerate(bary_coords):
            if j != ufc_facet:
                assert val > -1e-12


# -- Some useful helper methods --
def get_facet_from_bary_index(cell, bary_index):
    """
    Return the facet on which a given barycentric coordinate vanishes.
    
    For simplex cells, each barycentric coordinate is associate with a vertex v. The facet
    opposite v (which is also the facet NOT containing v) is exactly the facet on which the 
    corresponding barycentric coordinate vanishes.

    This function inverts the mapping:
        given a facet, identify the opposite vertex, deduce the vanishing barycentric coordinate.
    
    For tensor-product cells, this function is applied per axis
    (where each axis cell is a simplex, e.g. a UFCInterval).
    """
    facet_dim = cell.get_spatial_dimension() - 1
    topology = cell.get_topology()

    for facet, vertices in topology[facet_dim].items():
        if bary_index not in vertices:
            return facet

def get_bary_index_from_facet(cell, facet_index):
    """
    Return the index of the barycentric coordinate that vanishes on the given facet of a reference cell.

    This function is based on the following invariant:
        The barycentric coordinate associated with a vertex v vanishes exactly on the facet opposite to v.
    """
    # vertices of the full cell
    all_vertices = set(cell.get_topology()[0].keys())

    # vertices spanning the given facet
    facet_vertices = set(cell.get_topology()[cell.get_dimension() - 1][facet_index])

    # the unique vertex not in the facet
    missing = all_vertices - facet_vertices

    return next(iter(missing))






