import numpy as np
from firedrake import *
from ufl.differentiation import ReferenceGrad
from firedrake.interpolation import get_interpolator

mesh = UnitSquareMesh(5,5)
x = SpatialCoordinate(mesh)
invJ_expr = inv(ReferenceGrad(x))

coord = np.array([[0.5, 0.5]])
particle_vom = VertexOnlyMesh(mesh, coord)
X = particle_vom.reference_coordinates

U = VectorFunctionSpace(particle_vom, "DG", 0)
u = Function(U)
u.dat.data_wo[:] = np.array([0.15, 0.156])

# Evaluate the Forward Euler update expression
dt = 0.1

## This doesn't work since invJ_expr is not defined on the VOM
# update_expr = X + dt * invJ_expr * u
# new_coords = assemble(interpolate(update_expr, particle_vom.reference_coordinates.function_space()))


## Works if assembling invJ_expr into a Function on the VOM first as all terms in the update expression
# now become Functions on the VOM
T = TensorFunctionSpace(particle_vom, "DG", 0)
invJ_fn = Function(T)
invJ_fn.interpolate(invJ_expr)

breakpoint()

update_expr = X + dt * invJ_fn * u
new_coords = assemble(interpolate(update_expr, particle_vom.reference_coordinates.function_space()))

breakpoint()

## Keep the interpolation of invJ_expr * u onto VOM symbolic and evaluate it within the stepper interpolate
# Case 1: u is on VOM
# u_ref = dot(interpolate(invJ_expr, T), u)
u_ref = interpolate(invJ_expr, T) * u

# TODO: Currently not handled
# Case 2: u is not on VOM
# u_ref = interpolate(invJ_expr * u, U) # u is not on VOM

update_expr = X + dt * u_ref
new_coords = assemble(interpolate(update_expr, particle_vom.reference_coordinates.function_space()))

breakpoint()

## Cast u_ref as a mixed Function
# W = T * U # Mixed FS
# expr = as_vector([invJ_expr[i,j] for i in range(2) for j in range(2)] + [u[k] for k in range(2)])
# u_ref = interpolate(expr, W)
# mixed_interpolator = get_interpolator(u_ref) # MixedInterpolator
# sub_interpolators = mixed_interpolator._get_sub_interpolators(bcs=[])


