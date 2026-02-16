from firedrake import *
import numpy as np

mesh = UnitSquareMesh(10, 10, quadrilateral=False)

points = [[0.3, 0.6],
          [0.2, 0.8]]
vom = VertexOnlyMesh(mesh, points)

# interpolate a function defined on the VOM
V = FunctionSpace(mesh, "CG", 1)
v = Function(V)
asssemble(interpolate(v, vom.coordinates.function_space()))

