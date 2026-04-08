from firedrake import *
from firedrake.pyplot import triplot
import matplotlib.pyplot as plt
import numpy as np
from particle_traj_loop import move_particles_in_ref_space

# Define a flat reference mesh
mesh = UnitSquareMesh(10, 10)
# print("Mesh coords: ", mesh.coordinates.dat.data_ro[:5])

# Lift coordinates into a degree 2 space
# by define a non-linear map (s,t) -> (x, y) where
# (s,t) are the coords. of the flat mesh
# (x,y) are the coords. of the curved mesh
V = VectorFunctionSpace(mesh, "CG", 2)
v = Function(V)

s, t = SpatialCoordinate(mesh)

# x_new = s
# y_new = t + 0.2 * sin(pi*s)*sin(pi*t)

x_new = s + 0.5 *s*(1-s)
y_new = t + 0.5 *t*(1-t)

v.interpolate(
    as_vector([x_new, y_new])
)
curved_mesh = Mesh(v)
# print("Curved mesh coords: ", curved_mesh.coordinates.dat.data_ro[:5])

# fig, axes = plt.subplots()
# triplot(curved_mesh, axes=axes)
# axes.legend()
# plt.savefig(f"plots/curved_mesh.png", dpi=150)
# plt.close(fig)

# output = VTKFile(f"plots/curved_mesh.pvd")
# output.write(curved_mesh)

# Define the particle VOM
# Place 10 particles at midpoints of the diagonal cell edges: 0.05, 0.15, ..., 0.95
# These lie on y=x but avoid mesh vertices (which are at multiples of 0.1 for a 10x10 mesh)
n_particles = 10
x_diag = np.arange(n_particles) / 10.0 + 0.05
x0 = np.column_stack([x_diag, x_diag])
particle_vom = VertexOnlyMesh(curved_mesh, x0)
x0_vom = particle_vom.coordinates.dat.data_ro.copy()
print("Initial particle positions: ", particle_vom.coordinates.dat.data_ro)

# Assign per particle velocities
V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
v = Function(V, name="particle_velocity")
V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=particle_vom.geometric_dimension)
v_io = Function(V_io, name="io_particle_velocity")

v0 = np.array( [0.01, 0.02])
v_io.dat.data_wo[:] = np.tile(v0, (n_particles, 1))
v.interpolate(v_io)
v0_vom = v.dat.data_ro.copy()
print("Initial particle velocities: ", v0_vom)

T = 7.5
dt = 0.1
T_final, removed_particles = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0, plot=False)

print()
print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

x_final_expected = x0_vom + T_final * v0_vom

keep = np.ones(x_final_expected.shape[0], dtype=bool)
keep[removed_particles] = False
x_final_expected_survived = x_final_expected[keep]

print("Expected final positions: ", x_final_expected_survived)
print("Error: ", np.linalg.norm(x_final_expected_survived - particle_vom.coordinates.dat.data_ro, axis=1))
