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

# bary_coords_quad_sympy = quadrilateral.compute_barycentric_coordinates(p)
# bary_coorbds_quad_ufl = quadrilateral.compute_barycentric_coordinates(x_2d)

# bary_coords_interval_x_interval_sympy = interval_x_interval.compute_axis_barycentric_coordinates(p)
# bary_coords_interval_x_interval_ufl = interval_x_interval.compute_axis_barycentric_coordinates(x_2d)

# bary_coords_tri_x_interval_sympy = triangle_x_interval.compute_axis_barycentric_coordinates(q)
# bary_coords_tri_x_interval_ufl = triangle_x_interval.compute_axis_barycentric_coordinates(x_3d)

# Construct a GEM expression representing a point set
N = 1
dim = 2
# rt_X = gem.Variable("rt_X", shape=(N, dim)) 
# p = gem.Index("p", extent=N) # point index 
# i = gem.Index("i", extent=dim) # coordinate component index
# point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (p, i)),(i, )) # rank-1 tensor (vector) with free indices i and p and unindexed by component index i
# single_point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (0, i)),(i, )) # rank-1 tensor (vector) with only one free index i and unindexed by component index i (collapsed the point dim.)

rt_X = gem.Variable("rt_X", shape=(2, )) # no free indices

# NOTE: when `compute_barycentric_coordinates` performs numpy operations on its input `points` and we pass GEM nodes to it
# the output obtained is the result of doing numpy manipulations on GEM objects, not GEM tensor algebra operations
# so the GEM output is broken

# NEW: After changing numpy operations to matrix operations (which have a specialised implementation in GEM)
# and having GEM create a Literal node for the other numpy array operand in GEM.Node.__matmul__
bary_coords_tri_gem = triangle.compute_barycentric_coordinates(rt_X)
bary_coords_quad_gem  = quadrilateral.compute_barycentric_coordinates(rt_X)
bary_coords_tp_gem = triangle_x_interval.compute_axis_barycentric_coordinates(gem.Variable("rt_X", shape=(3, )))

breakpoint()

# Check the GEM expression by evaluating it at a given point
# This requires building/compiling a TSFC kernel
x_hat = np.array([0.25, 0.25])
x_hat_3d = np.array([0.25, 0.25, 0.25])

# Test finat.fiat_elements.barycentric_coordinates
# This generates a GEM expression for barycentric coordinates by converting SymPy to GEM
ref_element = mesh_2d.coordinates.function_space().finat_element
ref_element_quad = mesh_2d_quad.coordinates.function_space().finat_element

# -- Numeric tabulation in FIAT
bary_coords_tri_vals = ref_element.cell.compute_barycentric_coordinates(x_hat)
bary_coords_quad_vals = ref_element_quad.cell.compute_barycentric_coordinates(x_hat)

# Test a 3D TP cell
bary_coords_tp_vals = triangle_x_interval.compute_axis_barycentric_coordinates(x_hat_3d)

print("Barycentric coords tabulated by FIAT (triangle):", bary_coords_tri_vals)
print("Barycentric coords tabulated by FIAT (quad):", bary_coords_quad_vals)

print("Barycentric coords tabulated by FIAT (TP cell in 3D):", bary_coords_tp_vals)

# bary_cords_tri_gem_finat = ref_element.barycentric_coordinates(single_point_vec)
# NOTE: This doesn't work for quads since the FInAt element is a `FlattenedDimensions`
# which is not a FiatElement. 
# A possible solution to make this work is to implement a barycentric_coordinates() method in the FlattenedDimensions class

from gem_eval import evaluate_gem
# Evaluate GEM produced by FInAT (on FiatElement only)
# bary_coords_finat_gem_vals = evaluate_gem(bary_cords_tri_gem_finat, x_hat)
# print("Barycentric coords obtained by evaluating FInAt's GEM expr (triangle):", bary_coords_finat_gem_vals)

# Evaluate GEM produced by FIAT
bary_coords_fiat_gem_vals = evaluate_gem(bary_coords_tri_gem, x_hat)
bary_coords_gem_vals_quad = evaluate_gem(bary_coords_quad_gem, x_hat)

print("Barycentric coords obtained by evaluating FIAT's GEM expr (triangle):", bary_coords_tri_vals)
print("Barycentric coords obtained by evaluating FIAT's GEM expr (quad):", bary_coords_gem_vals_quad)

# NOTE: `bary_coords_tp_gem` is a list of GEM expressions representing barycentric coordinates along each axis
# of the TP cell. Hence we need to call `evaluate_gem`` on each of these expressions separately
bary_coords_gem_vals_tp_axis0 = evaluate_gem(bary_coords_tp_gem[0], x_hat_3d)
bary_coords_gem_vals_tp_axis1 = evaluate_gem(bary_coords_tp_gem[1], x_hat_3d)
print(f"Barycentric coords obtained by evaluating FIAT's GEM expr (TP cell in 3D):\n \
      axis 0: {bary_coords_gem_vals_tp_axis0},\n \
      axis 1: {bary_coords_gem_vals_tp_axis1}")

