from firedrake import *
import pytest
import numpy as np
from update_vom import VertexOnlyMeshUpdater

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
    """
    --- Tests ---
    - pick an interior facet
    - define a point on that facet (e.g., facet midpoint) and embedd into each adjacent cell
    - check that each embedded point lies on the correct facet of each cell 
    (using each cell's local facet ID and checking that the right barycentric coord. vanishes)
    - check that applying the backward transform on each embedded point results in the point embedded in the other cell
    - check that the embedded points map to the same physical point
    
    NOTE:
    To map the reference coordinates of a point back to its physical coordinates, we need to know which cell the point is contained in
    and do:

    \sum_i \hat_{x_i} * \Phi_i(X) where

    - \hat{x_i} are the global coordinates of the vertices of the cell that contains the point,
    - \Phi_i are the local nodal basis functions (for tensor-product cells, these would be bilinear/trilinear nodal basis functions)
    
    This basis evaluation is done automatically in Firedrake's point evaluation feature (see https://www.firedrakeproject.org/point-evaluation.html
    
    In our case, to obtain the physical coordinates from known reference coordinates, we do the following:
    - define a VOM containing a single point and overwrite its reference coordinates with the known reference coordinates 
        of the point we want to get the physical coordinates of.
    - interpolate the parent mesh coordinate field into this VOM.
    """
    import FIAT

    mesh = mesh_factory()
    topo = mesh.topology
    ref_cell = FIAT.ufc_cell(mesh.ufl_cell())
    
    if topo.interior_facets.set.total_size == 0:
        pytest.skip("Mesh has no interior facets")
    
    facet_cells = topo.interior_facets.facet_cell # adjacent cells
    local_facets = topo.interior_facets.local_facet_dat.data # local facet ID in each cell

    for f in range(facet_cells.shape[0]):
        c0, c1 = facet_cells[f]
        if c0 != -1 and c1 != -1:
            lf0, lf1 = local_facets[f]
            break
    
    facet_dim = topo.facet_dimension()

    # define the facet midpoint in facet-local coords.
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

    # embedd facet point onto each cell's reference frame
    A_embed, b_embed, _ = topo._get_facet_embedding_maps()
    X0_ref = A_embed[lf0] @ xi_facet + b_embed[lf0] # ref coords in cell 0
    X1_ref = A_embed[lf1] @ xi_facet + b_embed[lf1] # ref coords in cell 1
    
    # check that each embedded point lies on the correct facet of each cell
    bary0 = ref_cell.compute_barycentric_coordinates(X0_ref)
    bary1 = ref_cell.compute_barycentric_coordinates(X1_ref)

    tdim = mesh.topological_dimension
    if tdim == 1:
        # in 1D, facets are co-dim 1 vertices
        assert abs(bary0[lf0] - 1.0) < tol
        assert abs(bary1[lf1] - 1.0) < tol
    else:
        # in 2D/3D, facets are co-dim 1 faces
        assert abs(bary0[lf0]) < tol
        assert abs(bary1[lf1]) < tol

    # check coordinate maps in both directions
    X1_from0 = A01 @ X0_ref + b01
    X0_from1 = A10 @ X1_ref + b10

    assert np.allclose(X1_from0, X1_ref, atol=tol)
    assert np.allclose(X0_from1, X0_ref, atol=tol)

    # check that both embedded points correspond to the same physical point
    dummy_point = np.zeros((1, mesh.geometric_dimension))
    vom = VertexOnlyMesh(mesh, dummy_point)
    vom_updater = VertexOnlyMeshUpdater(vom, mesh)

    x0_phys = physical_point_from_reference(mesh, vom, vom_updater, X0_ref, c0)
    x1_phys = physical_point_from_reference(mesh, vom, vom_updater, X1_ref, c1)

    assert np.allclose(x0_phys, x1_phys, atol=tol)


def physical_point_from_reference(mesh, vom, vom_updater, cell_ref_point, parent_cell):
    # overwrite the parent cell and reference coordinates of the dummy point
    vom_updater.update_ref_view(
        next_parent_cells=[parent_cell],
        new_refcoords=[cell_ref_point]
    )
    x = assemble(interpolate(SpatialCoordinate(mesh), vom.coordinates.function_space()))
    return x.dat.data_ro[0]

# NOTE:
# - SimplicialComplex.compute_barycentric_coordinates returns a 1D array of shape (sd, ) for a single point 
# and a 2D array of shape (N, sd) for a batch of points
# - Hypercube.compute_barycentric_coordinates