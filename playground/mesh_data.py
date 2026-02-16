from firedrake import *
import numpy as np
import FIAT

mesh = UnitSquareMesh(10, 10, quadrilateral=True)
print(mesh.interior_facets.set.total_size)
ref_cell = FIAT.ufc_cell(mesh.ufl_cell())

# print(ref_cell.get_vertices())

# is_simplicial = isinstance(ref_cell, FIAT.reference_element.SimplicialComplex)
# print(is_simplicial)

# is_tensorproduct = isinstance(ref_cell, FIAT.reference_element.Hypercube)
# print(is_tensorproduct)

# tp_ref_cell = ref_cell.product
# print(tp_ref_cell)
# factors = tp_ref_cell.cells
# print(len(factors))
# print(factors[0])
# print(factors[0].get_spatial_dimension())

cell_coord_transforms_A, cell_coord_transforms_b = mesh.cell_facet_coord_transforms
print(cell_coord_transforms_A.data[0])
print(cell_coord_transforms_b.data[0])

# print(mesh.entity_orientations[0])
# print(mesh.interior_facets.local_facet_orientation_dat.data[0])
