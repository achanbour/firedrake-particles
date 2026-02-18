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
point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (p, i)),(i, ))
single_point_vec = gem.ComponentTensor(gem.Indexed(rt_X, (0, i)),(i, ))

# NOTE: meaningless output -- FIAT routines are not meant to accept GEM objects
# what we need is a method in FInAT that has a callback to fiat.reference_element.compute_barycentric_coordinates
bary_coords_tri_gem = triangle.compute_barycentric_coordinates(point_vec)

breakpoint()

# Test finat.fiat_elements.barycentric_coordinates
ref_element = mesh_2d.coordinates.function_space().finat_element
bary_coords_gem_expr_single_point = ref_element.barycentric_coordinates(single_point_vec)

# Check the GEM expression by evaluating it at a given point
x_hat = [[0.25, 0.25]]

# -- Numeric tabulation executed by FIAT
bary_coords_vals = ref_element.cell.compute_barycentric_coordinates(x_hat) 

# -- GEM is TSFC's IR so we evaluate the GEM expression by compiling/executing a TSFC kernel
import gem
import gem.impero_utils as impero_utils
from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder

# Create an output GEM variable
k = gem.Index("k", extent=3) # free index representing the component in the bary coords. vector
A = gem.Variable("A", shape=(3, )) # output 

# Create a rank-1 assignment and express it as an Impero program
# From tensor algebra (GEM) to scheduled tensor algebra (Impero)
return_expr = gem.Indexed(A, (k, ))
evaluation_expr = gem.Indexed(bary_coords_gem_expr_single_point, (k, ))
impero_c = impero_utils.compile_gem([(return_expr, evaluation_expr)], (k, ))

# Build a TSFC kernel
kernel_builder = ExpressionKernelBuilder("double")

# Collect kernel arguments
kernel_builder.set_coefficient_numbers(())
kernel_builder.set_coefficients([]) # no Function coefficients
kernel_builder.set_constants([]) # no constants
# -> only input to the kernel is the runtime coordinate buffer (rt_X); this is inferred below

kernel_builder.set_output(A) # pass the output variable A as argument
kernel_builder.register_requirements([evaluation_expr]) # infer other kernel arguments from the dependencies from the GEM evaluation expression tree (e.g., rt_X)

# Generate a Loopy kernel from the Impero program 
# the output is a TSFC ExpressionKernel that wraps the Loopy kernel
# kernel.ast is the Loopy TranslationUnit from which C code can be obtained
kernel = kernel_builder.construct_kernel(impero_c, {}, False, False)

# import loopy 
# ccode = loopy.generate_code_v2(kernel.ast) # string of generated C code 

"""
# C device code is the arithmetic code 
void expression_kernel(double* A, double const* rt_X)
{
  t0[0] = 1 - rt_X[0] - rt_X[1];
  t0[1] = rt_X[0];
  t0[2] = rt_X[1];
  for i:
      A[i] += t0[i];
}
"""

# Wrap the Loopy kernel for execution in PyOP2
from pyop2 import op2
from pyop2.local_kernel import LoopyLocalKernel
tu = kernel.ast # Loopy TranslationUnit (kernel IR)
lk = LoopyLocalKernel(tu, "expression_kernel") # runtime wrapper of the kernel IR (specifies the entrypoint kernel)

# Define an iteration set of size 1 as we only have one point
# This will run the expression evaluation kernel once
iterset = op2.Set(1)

# Provide a concrete runtime input point
rt_X_data = np.array([0.25, 0.25], dtype=float)
rt_X_global = op2.Global(2, data=rt_X_data)

# Allocate output array
A_out = np.zeros(3, dtype=float)
A_global = op2.Global(3, data=A_out)
#
# Execute the kerne
# JIT compiles the executable kernel into a shared library
# loading this shared library produces a pointer to a callable C function
# this function then gets called in a par_loop (only once here)
op2.par_loop(
    lk,
    iterset,
    A_global(op2.INC), # increment access (kernel computes increments to be summed into a global output object)
    rt_X_global(op2.READ), # read-only access
)

print("Barycentric coordinates computed by FIAT: ", bary_coords_vals)
print("Barycentric coordinates produced by evaluating GEM: ", A_out)


