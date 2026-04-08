from firedrake import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import numpy as np

# Define a flat reference mesh
mesh = UnitSquareMesh(5, 5) 
print("Mesh coords: ", mesh.coordinates.dat.data_ro[:5])

# Lift coordinates into a degree 2 space
V = VectorFunctionSpace(mesh, "CG", 5)
v = Function(V)
print("Degree 2 coords: ", v.dat.data_ro[:5])

s,t = SpatialCoordinate(mesh)

# NOTE: sine reaches its full amplitude at the midpoint (s,t) = (0.5, 0.5) giving us
# a much more visible curvature than using an exact degree 5 polynomial such as
# 2.0 * s*(1-s)*t*(1-t)*(s+t) which has a tiny max. value hence produces a negligible displacement
# (use high amplitude to counter suppression)
x_new = s
y_new = t + 0.3 * sin(pi * s) * sin(pi * t) # high amplitude can overshoot domain boundaries

v.interpolate(
    as_vector([x_new, y_new])
)

curved_mesh = Mesh(v)
print("Curved mesh coords: ", curved_mesh.coordinates.dat.data_ro[:5])

fig, axes = plt.subplots()
triplot(curved_mesh, axes=axes)
axes.legend()
plt.show()
