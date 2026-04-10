from firedrake import *
import numpy as np
from particle_traj_loop import move_particles_in_ref_space

"""
Deterministic particle trajectory test with constant velocity (identical for all particles)
and starting positions on the mesh diagonal.
"""

# Define the parent mesh
mesh = UnitSquareMesh(10, 10, quadrilateral=False)

# Define the particle VOM
# Place 10 particles at midpoints of the diagonal cell edges: 0.05, 0.15, ..., 0.95
# These lie on y=x but avoid mesh vertices (which are at multiples of 0.1 for a 10x10 mesh)
n_particles = 10
x_diag = np.arange(n_particles) / 10.0 + 0.05
x0 = np.column_stack([x_diag, x_diag])
particle_vom = VertexOnlyMesh(mesh, x0)
x0_vom = particle_vom.coordinates.dat.data_ro.copy()
print("Initial particle positions: ", x0_vom)

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

# dt=0.1, t=2.6: 1 particle removed
# dt=0.1, t=7.6: 2 particles removed
T = 7.6
dt = 0.1
T_final, removed_particles = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0, plot=False)

print()
print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

from particle_traj_loop import BISECTION_COUNT
print("Number of bisection calls to resolve crossings: ", BISECTION_COUNT)

x_final_expected = x0_vom + T_final * v0_vom

keep = np.ones(x_final_expected.shape[0], dtype=bool)
keep[removed_particles] = False
x_final_expected_survived = x_final_expected[keep]

print("Expected final positions: ", x_final_expected_survived)
print("Error: ", np.linalg.norm(x_final_expected_survived - particle_vom.coordinates.dat.data_ro, axis=1))
