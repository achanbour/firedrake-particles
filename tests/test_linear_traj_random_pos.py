from firedrake import *
from firedrake.petsc import PETSc
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
Linear particle trajectory using constant velocity (identical for all particles) and random initial positions.
"""

np.random.seed(42)

t = 0.0 # current time
dt = 0.01 # time step
T = 1.0

# Define the parent mesh
parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)

_ = parent_mesh.topology.cell_facet_neighbours
_ = parent_mesh.topology.cell_facet_coord_transforms

# Define the particles VOM
num_particles = 10
particle_coords = np.random.rand(num_particles, 2)
particle_vom = VertexOnlyMesh(parent_mesh, particle_coords)
x0_vom = particle_vom.coordinates.dat.data_ro.copy()
print("Initial particle positions (in VOM order): ", x0_vom)

# Define per-particle velocities
V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=particle_vom.geometric_dimension)
v = Function(V)
v_io = Function(V_io)
input_velocities = np.random.normal(0.1, 0.5, size=(num_particles,2))
v_io.dat.data[:] = input_velocities
v.interpolate(v_io)
v0_vom = v.dat.data_ro.copy()
print("Initial particle velocities: ", v)


# Define solvers
t_start = 0.0
t_end = 1
dt = 0.01
stepper = ForwardEulerStepper(particle_vom, dt, v=v)

bisection_params = BisectionSolverParams(max_iters=30)
cell_crossing_solver = BisectionSolver(bisection_params)

abs_time_tol = 1e-9
bary_tol = 1e-9
particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=bary_tol,
    abs_time_tol=abs_time_tol,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

with PETSc.Log.Event("ParticleTrajectoryLoop"):
    T_final, removed_particles = particle_traj_solver.solve(t_start, t_end)

print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

x_final_expected = x0_vom + T_final*v0_vom

keep = np.ones(x_final_expected.shape[0], dtype=bool)
keep[removed_particles] = False
x_final_expected_survived = x_final_expected[keep]

print("Exact final particle positions: ", x_final_expected_survived)
print("Error: ", np.linalg.norm(x_final_expected_survived - particle_vom.coordinates.dat.data_ro, axis=1))

# print(
#     f"Exact results matched up to bary_tol={bary_tol}: \
#     {np.allclose(particle_vom.coordinates.dat.data_ro, x_final_expected_survived, atol=bary_tol, rtol=0)}")

# print(
#     f"Exact results matched up to time_tol={abs_time_tol}: \
#     {np.allclose(particle_vom.coordinates.dat.data_ro, x_final_expected_survived, atol=abs_time_tol, rtol=0)}")