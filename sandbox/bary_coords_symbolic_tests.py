from firedrake import *
import numpy as np
import sympy as sp
import gem

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, TensorProductCell, UFCQuadrilateral, UFCHexahedron

interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = UFCQuadrilateral()
hexahedron = UFCHexahedron()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)

mesh_1d = UnitIntervalMesh(10)
mesh_2d = UnitSquareMesh(10, 10)
mesh_2d_quad = UnitSquareMesh(10, 10, quadrilateral=True)
mesh_3d = UnitCubeMesh(10, 10, 10)

# UFL symbolics (point on a mesh domain)
x_1d = SpatialCoordinate(mesh_1d)
x_2d = SpatialCoordinate(mesh_2d)
x_3d = SpatialCoordinate(mesh_3d)

# SymPy symbolics (free symbol)
x1, x2, x3 = sp.symbols("x1 x2 x3")
p = [x1, x2]
q = [x1, x2, x3]

# Test FIAT.reference_element.compute_barycentric_coordinates on a symbolic input
bary_coords_interval_sympy = interval.compute_barycentric_coordinates([x1])
bary_coords_interval_ufl = interval.compute_barycentric_coordinates(x_1d)

bary_coords_tri_sympy = triangle.compute_barycentric_coordinates(p)
bary_coords_tri_ufl = triangle.compute_barycentric_coordinates(x_2d)

bary_coords_quad_sympy = quadrilateral.compute_barycentric_coordinates(p)
bary_coorbds_quad_ufl = quadrilateral.compute_barycentric_coordinates(x_2d)

bary_coords_interval_x_interval_sympy = interval_x_interval.compute_axis_barycentric_coordinates(p)
bary_coords_interval_x_interval_ufl = interval_x_interval.compute_axis_barycentric_coordinates(x_2d)

bary_coords_tri_x_interval_sympy = triangle_x_interval.compute_axis_barycentric_coordinates(q)
bary_coords_tri_x_interval_ufl = triangle_x_interval.compute_axis_barycentric_coordinates(x_3d)

breakpoint()
# Test fiat.reference_element.compute_barycentric_coordinates on a GEM input

# Construct a GEM expression for a point set in a 2D mesh
N = 1
dim = 2
p = gem.Index("p", extent=N) # point index 
i = gem.Index("i", extent=dim) # coordinate component index

rt_X = gem.Variable("rt_X", shape=(N, dim)) 
point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (p, i)),(i, )) # rank-1 tensor indexed by i, parametrised by free point index p
single_point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (0, i)),(i, )) # rank-1 tensor indexed by i (no point index p)

# NOTE: meaningless output -- FIAT routines are not meant to accept GEM objects
# what we need is a method in FInAT that has a callback to fiat.reference_element.compute_barycentric_coordinates
bary_coords_tri_gem = triangle.compute_barycentric_coordinates(point_vec)

breakpoint()

# Test finat.fiat_elements.barycentric_coordinates

ref_element = mesh_2d.coordinates.function_space().finat_element
ref_element_quad = mesh_2d_quad.coordinates.function_space().finat_element

# Check the GEM expression by evaluating it at a given point
x_hat = [[0.25, 0.25]]

# -- Numeric tabulation executed by FIAT
bary_coords_vals = ref_element.cell.compute_barycentric_coordinates(x_hat)
bary_coords_vals_quad = ref_element_quad.cell.compute_barycentric_coordinates(x_hat)

print("Barycentric coords computed by FIAT (triangle):", bary_coords_vals)
# print("Barycentric coords computed by FIAT (quad):", bary_coords_vals_quad)

# -- GEM is TSFC's IR so we evaluate the GEM expression by compiling/executing a TSFC kernel
bary_gem_expr = ref_element.barycentric_coordinates(single_point_vec)

# NOTE: Doesn't work for quads since the FInAt element is a `FlattenedDimensions`
# which is not a FiatElement. A possible solution to make this work is to implement a barycentric_coordinates() method
# in the FlattenedDimensions class
# bary_gem_expr_quad = ref_element_quad.barycentric_coordinates(single_point_vec)

from gem_eval import evaluate_gem
bary_coords_gem_vals = evaluate_gem(bary_gem_expr, x_hat)
# bary_coords_gem_vals_quad = evaluate_gem(bary_gem_expr_quad, x_hat)

print("Barycentric coords obtained by evaluating GEM (triangle):", bary_coords_gem_vals)
# print("Barycentric coords evalauted by evaluating GEM (quad):", bary_coords_gem_vals_quad)



