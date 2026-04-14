import os
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import triplot, pointplot

os.makedirs("pointplot_output", exist_ok=True)
np.random.seed(42)

# 1. Plot particles positions
mesh = UnitSquareMesh(10, 10)
coords = np.random.rand(50, 2)
vom = VertexOnlyMesh(mesh, coords)

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
pointplot(vom, axes=axes)
axes.set_aspect("equal")
axes.set_title("Particle positions")
# axes.legend()
plt.savefig("pointplot_output/demo_1_basic.png", dpi=150)
plt.close(fig)


# 2. Plot particles coloured by a scalar field
V = FunctionSpace(vom, "DG", 0)
f = Function(V)
f.dat.data_wo[:] = vom.coordinates.dat.data_ro[:, 0] # colour by x-coordinate

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
sc = pointplot(vom, function=f, axes=axes, cmap="plasma")
fig.colorbar(sc, ax=axes, label="x-coordinate")
axes.set_aspect("equal")
axes.set_title("Particles coloured by scalar field")
plt.savefig("pointplot_output/demo_2_scalar_field.png", dpi=150)
plt.close(fig)


# 3. Layared plot: plot particles on top of a PDE solution (solved on the parent mesh)
from firedrake.pyplot import tripcolor
V_mesh = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
u = Function(V_mesh)
u.interpolate(sin(pi*x)*sin(pi*y))

fig, axes = plt.subplots()
tripcolor(u, axes=axes, cmap="viridis")
pointplot(vom, axes=axes, c="white", edgecolors="black", s=20)
axes.set_aspect("equal")
axes.set_title("Particles on PDE solution")
plt.savefig("pointplot_output/demo_3_layered.png")
plt.close(fig)




