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

# UFL symbolics (point on a mesh domain)
mesh_1d = UnitIntervalMesh(10)
mesh_2d = UnitSquareMesh(10, 10)
mesh_2d_quad = UnitSquareMesh(10, 10, quadrilateral=True)
mesh_3d = UnitCubeMesh(10, 10, 10)

x_1d = SpatialCoordinate(mesh_1d)
x_2d = SpatialCoordinate(mesh_2d)
x_3d = SpatialCoordinate(mesh_3d)

# SymPy symbolics (free symbol)
x1, x2, x3 = sp.symbols("x1 x2 x3")
p = [x1, x2]
q = [x1, x2, x3]

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

# Construct a GEM expression for a point set
N = 1
dim = 2
p = gem.Index("p", extent=N) # point index p 
i = gem.Index("i", extent=dim) # coordinate component i 

rt_X = gem.Variable("rt_X", shape=(N, dim))
point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (p, i)),(i, ))
single_point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (0, i)),(i, ))

# NOTE: meaningless output -- FIAT routines are not meant to accept GEM objects
# what we need is a method in FInAT that has a callback to FIAT compute_barycentric_coordinates
bary_coords_tri_gem = triangle.compute_barycentric_coordinates(point_vec)

breakpoint()
ref_element = mesh_2d.coordinates.function_space().finat_element
bary_coords_gem_expr = ref_element.barycentric_coordinates(point_vec)
bary_coords_gem_expr_single_point = ref_element.barycentric_coordinates(single_point_vec)

# Check the GEM expression by evaluating it at a given point
x_hat = [[0.25, 0.25]]
# -- Numeric tabulation executed by FIAT returns the barycentric coords.
bary_coords_vals = ref_element.cell.compute_barycentric_coordinates(x_hat) 
# --GEM is TSFC's IR so we evaluate the GEM expression 
# by compiling/executing a TSFC kernel
import gem
import gem.impero_utils as impero_utils
from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder as interface

# Create an output GEM variable
k = gem.Index("k", extent=3)
A = gem.Variable("A", shape=(3, ))
return_expr = gem.Indexed(A, (k, ))
evaluation_expr = gem.Indexed(bary_coords_gem_expr_single_point, (k, ))
impero_c = impero_utils.compile_gem([(return_expr, evaluation_expr)], (k, ))

builder = interface("double")

# minimal initialization
builder.set_coefficient_numbers(())
builder.set_coefficients([])
builder.set_constants([])

builder.set_output(A)
builder.register_requirements([evaluation_expr])
kernel = builder.construct_kernel(impero_c, {}, False, False)

# import loopy 
# code = loopy.generate_code_v2(kernel.ast) # string of generated C code 

from pyop2 import op2
from pyop2.local_kernel import LoopyLocalKernel
tu = kernel.ast # TranslationUnit from ExpressionKernel

lk = LoopyLocalKernel(tu, "expression_kernel") # Wrap as a PyOP2 local kernel

# Iteration set of size 1 (one "point")
iterset = op2.Set(1)

# Input: rt_X is length-2
rt_X_data = np.array([0.25, 0.25], dtype=float)
rt_X_global = op2.Global(2, data=rt_X_data)

# Output: A is length-3
A_out = np.zeros(3, dtype=float)
A_global = op2.Global(3, data=A_out)

# Run the kernel once
op2.par_loop(
    lk,
    iterset,
    A_global(op2.INC),
    rt_X_global(op2.READ),
)

print("Executed barycentrics:", A_out)


