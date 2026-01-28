from firedrake import *
import numpy as np


def build_barycentric_facet_map_for_tensor_ref_cell(ref_cell, tol=1e-12):
    """
    Return a mapping from cell barycentric coordinates to cell facet ID of a tensor-product reference cell (e.g., UFCQuadrilateral).
    """
    coord_to_facet = {}

    facet_dim = ref_cell.get_spatial_dimension() - 1
    tp_cell = ref_cell.product
    axes = tp_cell.cells

    # Track where each axis' barycentric coordinates start in the stacked vector
    axes_offsets = []
    offset = 0
    for a in axes:
        axes_offsets.append(offset)
        offset += a.get_dimension() + 1

    # Slices for extracting sub-coordinates
    slices = tp_cell._split_slices([ax.get_dimension() for ax in axes])

    for facet_id in range(len(ref_cell.get_topology()[facet_dim])):
        # Define a point on the facet and get its reference coordinates in the full cell
        phi = ref_cell.get_entity_transform(facet_dim, facet_id) # mapping from the facet local coords to the full cell coords
        midpoint = np.full((1, facet_dim), 0.5) 
        mapped = phi(midpoint)[0]

        # For each axis, compute which barycentric coordinate vanishes at the mapped point
        for i, (axis, slice) in enumerate(zip(axes, slices)):
            axis_coords = mapped[slice] # extract ref coordinates along this axis
            lambdas = axis.compute_barycentric_coordinates(axis_coords.reshape(1, -1))[0]

            for local_idx, val in enumerate(lambdas):
                if abs(val) < tol:
                    global_index = axes_offsets[i] + local_idx
                    coord_to_facet[global_index] = facet_id
                    break
                
    return coord_to_facet


quad_mesh = UnitSquareMesh(10, 10, quadrilateral=True)

N = 10
seed = 42
np.random.seed(seed)

particle_coords = np.random.rand(N, 2)
vom = VertexOnlyMesh(quad_mesh, particle_coords)

ref_cell = quad_mesh.coordinates.function_space().finat_element.cell
# NOTE: see methods defined on the local topology in venv/lib/python/site-packages/FIAT/reference_element.py

# ref_cell.get_vertices()
# ref_cell.get_topology()
# ref_cell.compute_barycentric_coordinates() # not implemented for tensor-product cells

# ref_cell is a UFCQuadrilateral which is a child of UFCHypercube 
# which wraps a TensorProductCell in its `product` attribute
tp_cell = ref_cell.product

# each factor is an interval element (UFCInterval)
factors = tp_cell.cells 

breakpoint()

coord_to_facet_map = build_barycentric_facet_map_for_tensor_ref_cell(ref_cell)
print("Coord to facet map for quad cell: ", coord_to_facet_map)


    