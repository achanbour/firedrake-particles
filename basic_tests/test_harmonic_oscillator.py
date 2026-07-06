from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper_new import StormerVerletStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
This examples solves for the trajectory of particles subject to an external force derived
from a harmonic potential.
"""

# Set up
# -- Define the initial conditions
num_particles = 10
r = 0.25 # radius
c = np.array([0.5, 0.5]) # center
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)

x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)
particle_vom = VertexOnlyMesh(parent_mesh, q0)


# -- Define the potential field (on the parent mesh)
# Harmonic potential is V(q) = 0.5 * omega**2 * (x-c)**2
# The acceleration a is defined as a = -V'(q)
omega = 0.5
x = SpatialCoordinate(parent_mesh)
accel_expr = -omega**2*as_vector([x[0]-c[0], x[1]-c[1]])

# initial tangential velocity
init_pos = particle_vom.coordinates.dat.data_ro.copy()
V0 = omega * np.column_stack([-(init_pos[:, 1] - c[1]), init_pos[:, 0] - c[0]])

# Define the solvers
t_start = 0
t_end = 1
dt = 0.01

stepper = StormerVerletStepper(particle_vom, dt, accel_expr)
# assign initial velocity
stepper.v.dat.data_wo[:] = V0

cell_crossing_solver = BisectionSolver()
particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

T_final, surviving_particle_ids = particle_traj_solver.solve(t_start, t_end)

final_pos = particle_vom.coordinates.dat.data_ro

assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

print("Final particle positions:", final_pos)

# --- exact solution: trajectory is circular at angular rate omega
# same as in steady rigid-body rotation case
a = omega * t_end
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
exact_counterclockwise = c + (init_pos - c) @ R.T
# exact_clockwise = c + (init_pos - c) @ R

# Check 3: numerical error O(dt^2)
print("max position error vs exact counterclockwise:", np.max(np.linalg.norm(final_pos - exact_counterclockwise, axis=1)))
# print("max position error vs exact clockwise:", np.max(np.linalg.norm(final_pos - exact_clockwise, axis=1)))

