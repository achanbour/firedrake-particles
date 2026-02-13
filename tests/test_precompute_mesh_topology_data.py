from firedrake import *
import pytest
import numpy as np


def interval_mesh():
    return UnitIntervalMesh(2)

def triangle_mesh():
    return UnitSquareMesh(1, 1)

def quad_mesh():
    return UnitSquareMesh(2, 1, quadrilateral=True)

def tet_mesh():
    return UnitCubeMesh(1, 1, 1)

def hex_mesh():
    return UnitCubeMesh(2, 1, 1, hexahedral=True)

@pytest.mark.parametrize(
    "mesh_factory",
    [
        pytest.param(interval_mesh, id="interval"),
        pytest.param(triangle_mesh, id="triangle"),
        pytest.param(quad_mesh, id="quadrilateral"),
        pytest.param(tet_mesh, id="tetrahedron"),
        pytest.param(hex_mesh, id="hexahedron"),
    ]
)
def test_cell_facet_coord_transforms(mesh_factory, tol=1e-12):
    import FIAT

    mesh = mesh_factory()
    topo = mesh.topology
    ref_cell = FIAT.ufc_cell(mesh.ufl_cell())
    
    if topo.interior_facets.set.total_size == 0:
        pytest.skip("Mesh has no interior facets")
    
    # --- Tests ---
    # pick an interior facet
    # embedd midpoint into each adjacent cell
    # check embedded point lies on the facet of each cell
    # check that applying backward transform results in the same point 
    # TODO: check embedded points map to the same physical point
    # \sum_i \hat_{x_i} * \Phi_i(X)
    # \hat{x_i} are the global coordinates of the vertices of the cell that the point is contained in
    # \Phi_i are the local nodal basis functions (for tensor-product cells, these would be bilinear/trilinear nodal basis functions)
    
    facet_cells = topo.interior_facets.facet_cell # adjacent cells
    local_facets = topo.interior_facets.local_facet_dat.data # local facet ID in each cell

    for f in range(facet_cells.shape[0]):
        c0, c1 = facet_cells[f]
        if c0 != -1 and c1 != -1:
            lf0, lf1 = local_facets[f]
            break
    
    facet_dim = topo.facet_dimension()
    # midpoint in facet-local coords.
    if facet_dim == 0:
        xi_facet = np.zeros(0)
    else:
        xi_facet = np.full(facet_dim, 0.5)

    A_dat, b_dat = topo.cell_facet_coord_transforms

    # c0 -> c1
    A01 = A_dat.data[c0, lf0]
    b01 = b_dat.data[c0, lf0]

    # c1 -> c0
    A10 = A_dat.data[c1, lf1]
    b10 = b_dat.data[c1, lf1]

    # embedd facet point onto each cell reference frame
    A_embed, b_embed, _ = topo._get_facet_embedding_maps()

    X0_ref = A_embed[lf0] @ xi_facet + b_embed[lf0] # ref coords in cell 0
    X1_ref = A_embed[lf1] @ xi_facet + b_embed[lf1] # ref coords in cell 1

    bary0 = ref_cell.compute_barycentric_coordinates(X0_ref)[0]
    bary1 = ref_cell.compute_barycentric_coordinates(X1_ref)[0]

    # check vanishing on facet
    tdim = mesh.topological_dimension
    if tdim == 1:
        # in 1D, facets are co-dim 1 vertices
        assert abs(bary0[lf0] - 1.0) < tol
        assert abs(bary1[lf1] - 1.0) < tol
    else:
        # in 2D/3D, facets are co-dim 1 faces
        assert abs(bary0[lf0]) < tol
        assert abs(bary1[lf1]) < tol

    X1_from0 = A01 @ X0_ref + b01
    X0_from1 = A10 @ X1_ref + b10

    # check maps in both directions
    assert np.allclose(X1_from0, X1_ref, atol=tol)
    assert np.allclose(X0_from1, X0_ref, atol=tol)



















    

