from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
Linear particle trajectory using constant velocity (identical for all particles) and initial positions on the mesh diagonal.
"""
# Define the parent mesh
parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)

# Define the particle VOM
n_particles = 10
x_diag = np.arange(n_particles) / 10.0 + 0.05
x0 = np.column_stack([x_diag, x_diag])
particle_vom = VertexOnlyMesh(parent_mesh, x0)
x0_vom = particle_vom.coordinates.dat.data_ro.copy()
print("Initial particle positions: ", x0_vom)

# Define per-particle velocities
V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
v = Function(V, name="particle_velocity")
V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=particle_vom.geometric_dimension)
v_io = Function(V_io, name="io_particle_velocity")

v0 = np.array( [0.01, 0.02])
v_io.dat.data_wo[:] = np.tile(v0, (n_particles, 1))
v.interpolate(v_io)
v0_vom = v.dat.data_ro.copy()
print("Initial particle velocities: ", v0_vom)

# Define solvers
# dt=0.1, t_end=2.6: 1 particle removed
# dt=0.1, t_end=7.6: 2 particles removed

t_start = 0
t_end = 2.6
dt = 0.1
bisection_params = BisectionSolverParams(max_iters=30)
particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)

stepper = ForwardEulerStepper(particle_vom, dt, v=v)
cell_crossing_solver = BisectionSolver(bisection_params)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

T_final, removed_particles = particle_traj_solver.solve(t_start, t_end)

print()
print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

x_final_expected = x0_vom + T_final * v0_vom

keep = np.ones(x_final_expected.shape[0], dtype=bool)
keep[removed_particles] = False
x_final_expected_survived = x_final_expected[keep]

print("Expected final positions: ", x_final_expected_survived)
print("Error: ", np.linalg.norm(x_final_expected_survived - particle_vom.coordinates.dat.data_ro, axis=1))
