from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
Circular particle trajectory using a solid body rotation field with constant angular speed
and starting positions aligned on a circle centered at c with radius r.
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
print(np.allclose(radii, r))

# Define the parent mesh
parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)

# Define the particles VOM
particle_vom = VertexOnlyMesh(parent_mesh, q0)
print("Initial particle positions: ", particle_vom.coordinates.dat.data_ro)

# Define the velocity field (on the parent mesh)
# v(q) = omega * J * q -> linear in space so use CG1 FS
omega = 0.5 # angular speed
x = SpatialCoordinate(parent_mesh)
v_expr = omega * as_vector([-x[1]+c[1], x[0]-c[0]])

# Define the solvers
t_start = 0
t_end = 1
dt = 0.01
stepper = ForwardEulerStepper(particle_vom, dt, v_expr)

cell_crossing_solver = BisectionSolver()

particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=True
)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

T_final, removed_particles = particle_traj_solver.solve(t_start, t_end)

print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

from particle_crossing_solver import BISECTION_COUNT
print("Number of bisection calls to resolve cell crossings: ", BISECTION_COUNT)

# NOTE: Forward Euler is not exact in this case.
# Visually, we can see particles moving away from the center of the circle they started on.
# This drift is due to the fact that, under Forward Euler, the radius ||q^n - c||^2 
# grows by a factor O(1+dt^2*omega*2) at each step.
# For more accurate results, use a structure-preserving scheme such as the implicit midpoint rule.








