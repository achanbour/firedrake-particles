from firedrake import *
import numpy as np
from ufl.differentiation import ReferenceGrad

mesh = UnitSquareMesh(30, 30, quadrilateral=False)
x = SpatialCoordinate(mesh)
invJ_expr = inv(ReferenceGrad(x)) # from phys -> ref

N = 10
coords = np.random.rand(N, 2)
vom = VertexOnlyMesh(mesh, coords)
TS = TensorFunctionSpace(vom, "DG", 0)
invJ_vom = Function(TS)

breakpoint()
invJ_vom.interpolate(invJ_expr)


