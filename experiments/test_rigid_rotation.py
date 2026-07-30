from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
Circular particle trajectory using a rigid-body rotation field with time-varying angular speed.
"""

# Params
num_particles = 10
r = 0.25 # radius
c = np.array([0.5, 0.5]) # center
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False) # initial angles evenly spaced on [0, 2pi)

# Initial positions: points lying on a circle of radius r centered at c
x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

radii = np.sqrt((q0[:, 0] - c[0])**2 + (q0[:, 1] - c[1])**2)
assert np.allclose(radii, r)

# Define the parent mesh
parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)

# Define the particles VOM
particle_vom = VertexOnlyMesh(parent_mesh, q0)
print("Initial particle positions: ", particle_vom.coordinates.dat.data_ro)

# Define the time-varying velocity field
omega0 = 0.5
omega = lambda t: omega0 * (1 + t) # time-varying angular speed
Theta = lambda t: omega0 * (t + 0.5*t**2) # accumulated angle 

x = SpatialCoordinate(parent_mesh)
rot = as_vector([-x[1]+c[1], x[0]-c[0]]) # unit rotation
V = VectorFunctionSpace(parent_mesh, "CG", 1)
v = Function(V, name="velocity")

# Define the solvers
time = 0 # Initial time
dt = 0.01 # Time step
timesteps = 100 # Maximum number of timesteps

stepper = ForwardEulerStepper(particle_vom, dt, v) # -- pass the velocity function
cell_crossing_solver = BisectionSolver()
particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

init_pos = particle_vom.coordinates.dat.data_ro.copy()

for time_step in range(0, timesteps):
    time_start = time

    # -- the changes in v must be picked up by the particle time stepper
    v.interpolate(omega(time)*rot)
    time += dt
    particle_traj_solver.solve(time_start, time)

final_pos = particle_vom.coordinates.dat.data_ro

from particle_crossing_solver import BISECTION_COUNT
print("Number of bisection calls to resolve cell crossings: ", BISECTION_COUNT)

# --- exact solution ---
t_end = dt * timesteps
a = Theta(t_end)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
exact = c + (init_pos - c) @ R.T

# Check that the angle has advanced by Theta(t_end)
ang0 = np.arctan2(init_pos[:,1]-c[1], init_pos[:,0]-c[0])
angN = np.arctan2(final_pos[:,1]-c[1], final_pos[:,0]-c[0])
dang = np.mod(angN - ang0, 2*np.pi)
assert np.allclose(dang, np.mod(Theta(t_end), 2*np.pi), atol=5e-3)


# Check numerical error is O(dt)
print("max position error vs exact:", np.max(np.linalg.norm(final_pos - exact, axis=1)))