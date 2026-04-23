import os
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import triplot, pointplot, quiver

os.makedirs("demo_vomplots", exist_ok=True)
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
plt.savefig("demo_vomplots/demo_1_basic.png", dpi=150)
plt.close(fig)

# 2. Plot particles coloured by a scalar field
V = FunctionSpace(vom, "DG", 0)
f = Function(V)
f.dat.data_wo[:] = vom.coordinates.dat.data_ro[:, 0] # colour by x-coordinate

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
sc = pointplot(f, axes=axes, cmap="plasma")
fig.colorbar(sc, ax=axes, label="x-coordinate")
axes.set_aspect("equal")
axes.set_title("Particles coloured by a scalar field")
plt.savefig("demo_vomplots/demo_2_scalar_field.png", dpi=150)
plt.close(fig)


# 3. Layered plot: plot particles on top of a PDE solution (solved on the parent mesh)
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
plt.savefig("demo_vomplots/demo_3_layered.png")
plt.close(fig)

# 4. 3D Plot: particles inside a unit cube
mesh_3d = UnitCubeMesh(5, 5, 5)
coords_3d = np.random.rand(30, 3)
vom_3d = VertexOnlyMesh(mesh_3d, coords_3d)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
triplot(mesh_3d, axes=axes, interior_kw={'alpha': 0.05}, boundary_kw={'alpha': 0.1})
pointplot(vom_3d, axes=axes, s=40, depthshade=False)
axes.set_title("3D particle positions")
axes.set_aspect("equal")
plt.savefig("demo_vomplots/demo_4_3d.png", dpi=150)
plt.close(fig)

# 5. Plot a vector field on particles
V_vec = VectorFunctionSpace(vom, "DG", 0, dim=2)
v = Function(V_vec)
v.dat.data_wo[:] = vom.coordinates.dat.data_ro - 0.5 # point away from the centre at (0.5, 0.5)

fig, axes = plt.subplots()
triplot(mesh, axes=axes)
pointplot(vom, axes=axes)
quiver(v, axes=axes)
axes.set_title("Particles with a vector field")
plt.savefig("demo_vomplots/demo_5_vector_field.png", dpi=150)
plt.close(fig)

