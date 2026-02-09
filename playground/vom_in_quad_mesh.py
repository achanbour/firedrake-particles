from firedrake import *
import numpy as np
from FIAT.reference_element import TensorProductCell, UFCInterval, UFCQuadrilateral, UFCHexahedron

# NOTE: The below function is wrong as it assumes a vertex-based indexing of barycentric coordinates in
# tensor-product cells. This only holds for simplices. In tensor-product cells, barycentric coordinates are facet-aligned.

def build_bary_coord_to_facet_map(ref_cell, tol=1e-12):
    """
    Return a mapping from cell barycentric coordinates to cell facet ID.
    """
    coord_to_facet = {}
    facet_dim = ref_cell.get_spatial_dimension() - 1
    tp_cell = ref_cell.product

    for facet_id in range(len(ref_cell.get_topology()[facet_dim])):
        phi = ref_cell.get_entity_transform(facet_dim, facet_id) # transform that maps coords. from facet to cell
        midpoint = np.full((1, facet_dim), 0.5) # define a point in the facet's coords. system
        mapped_midpoint = phi(midpoint)[0] # get the point in the cell's coords. system

        bary_coords = tp_cell.compute_barycentric_coordinates(mapped_midpoint.reshape(1, -1))[0]

        for coord_idx, val in enumerate(bary_coords):
            if abs(val) < tol:
                # record which barycentric coord vanishes
                coord_to_facet[coord_idx] = facet_id
                break

    return coord_to_facet

if __name__ == "__main__":
    quad_mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    N = 10
    seed = 42
    np.random.seed(seed)
    particle_coords = np.random.rand(N, 2)
    vom = VertexOnlyMesh(quad_mesh, particle_coords)
    ref_cell = quad_mesh.coordinates.function_space().finat_element.cell
    
    # NOTE: see methods defined on the local topology in venv/lib/python/site-packages/FIAT/reference_element.py
    
    ref_cell.get_vertices()
    ref_cell.get_topology()
    ref_cell.compute_barycentric_coordinates() # not implemented for tensor-product cells

    # ref_cell is a UFCQuadrilateral which is a child of UFCHypercube 
    # which wraps a TensorProductCell in its `product` attribute
    tp_cell = ref_cell.product

    # UFCQuadrilateral = UFCInterval x UFCInterval
    factors = tp_cell.cells 

    build_bary_coord_to_facet_map(ref_cell)


    