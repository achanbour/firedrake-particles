from firedrake import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import numpy as np
from particle_traj_loop import move_particles_in_ref_space

# Define a flat reference mesh
mesh = UnitSquareMesh(10, 10)
print("Mesh coords: ", mesh.coordinates.dat.data_ro[:5])

# Lift coordinates into a degree 2 space
V = VectorFunctionSpace(mesh, "CG", 2)
v = Function(V)

## Define a non-linear map (s,t) -> (x, y)
# (s,t) are the coords. of the flat mesh
# (x,y) are the coords. of the curved mesh
s, t = SpatialCoordinate(mesh)

# x_new = s
# y_new = t + 0.2 * sin(pi*s)*sin(pi*t)

x_new = s + 0.5 *s*(1-s)
y_new = t + 0.5 *t*(1-t)

v.interpolate(
    as_vector([x_new, y_new])
)
curved_mesh = Mesh(v)
print("Curved mesh coords: ", curved_mesh.coordinates.dat.data_ro[:5])

fig, axes = plt.subplots()
triplot(curved_mesh, axes=axes)
axes.legend()
plt.savefig(f"plots/curved_mesh.png", dpi=150)
plt.close(fig)

output = VTKFile(f"plots/curved_mesh.pvd")
output.write(curved_mesh)