from firedrake import *
import numpy as np

mesh = UnitSquareMesh(10, 10, quadrilateral=False)

points = [[0.3, 0.6],
          [0.2, 0.8]]

vom = VertexOnlyMesh(mesh, points)

# NOTE:
# The dual evaluation expression is generated during instantiation of the VertexOnlyMesh 
# The PS supplied to the DualEvaluationCallable are the target element's dual basis points
# which are the reference coordinates of the VOM points (in the parent mesh reference cell)

# interpolate(SpatialCoordinate(mesh), vom.coordinates.function_space())

breakpoint()
# To get an unknown PS, we need evaluation points that are not part of an element's dual basis.
V = FunctionSpace(mesh, "CG", 2)
# x,y = SpatialCoordinate(mesh)
# f = assemble(interpolate(x**2 + y**2, V))
f = Function(V)
P0DG = FunctionSpace(vom, "DG", 0)
f_at_points = assemble(interpolate(f, P0DG))

