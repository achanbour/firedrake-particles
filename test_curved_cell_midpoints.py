from firedrake import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import numpy as np

# Define a flat reference mesh
mesh = UnitSquareMesh(10, 10)
h = 1 / 10
print("Mesh coords: ", mesh.coordinates.dat.data_ro[:5])

# mesh.coordinates is a CG1 Function where nodes/DoFs match the cell vertices
vertex_coords = set(map(tuple, mesh.coordinates.dat.data_ro))

# Lift coordinates into a degree 2 space
V = VectorFunctionSpace(mesh, "CG", 2)
v = Function(V)
v.interpolate(mesh.coordinates)
# v contains midpoint DoFs in addition to the vertex DoFs
print("Degree 2 coords: ", v.dat.data_ro[:5])

perp = np.array([1.0, 1.0]) / np.sqrt(2)

# a global map F(s,t) -> (x,y) moves all nodes including vertices
# however, to only curve diagonal edges, we need to sample midpoint nodes on the diagonal
for i, (x,y) in enumerate(v.dat.data_ro):
    if tuple([x, y]) not in vertex_coords:
        # pick midpoints that are on the diagonal
        on_horizontal = np.isclose(y % h, 0) or np.isclose(y % h, h)
        on_vertical = np.isclose(x % h, 0) or np.isclose(y % h, h)
        if not on_horizontal and not on_vertical:
            # apply a displacement
            v.dat.data[i, 1] += 0.02

curved_mesh = Mesh(v)
print("Curved mesh coords: ", curved_mesh.coordinates.dat.data_ro[:5])

fig, axes = plt.subplots()
triplot(curved_mesh, axes=axes)
axes.legend()
plt.show()

output = VTKFile(f"plots/curved_midpoints.pvd")
output.write(curved_mesh)