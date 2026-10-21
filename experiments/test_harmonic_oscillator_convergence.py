from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

from particle_time_stepper import StormerVerletStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
This examples solves for the trajectory of particles subject to an external force derived
from a harmonic potential.
"""

# Initial conditions
num_particles = 10
r = 0.25 # radius
c = np.array([0.5, 0.5]) # center
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)

x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

# Solver params
t_start = 0
t_end = 0.2


errors = []
dts = [4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4]

for dt in dts:

    # -- Define the geometry
    parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)
    particle_vom = VertexOnlyMesh(parent_mesh, q0)


    # -- Define the potential field
    # Harmonic potential is V(q) = 0.5 * omega**2 * (x-c)**2
    # The acceleration a is defined as a = -V'(q)
    omega = 10*np.pi
    x = SpatialCoordinate(parent_mesh)
    accel_expr = -omega**2*as_vector([x[0]-c[0], x[1]-c[1]])

    Va = VectorFunctionSpace(parent_mesh, "CG", 1)
    accel_fn = Function(Va, name="acceleration").interpolate(accel_expr)

    # -- Initial velocity (tangential so that motion is circular)
    init_pos = particle_vom.coordinates.dat.data_ro.copy()
    V0 = omega * np.column_stack([-(init_pos[:, 1] - c[1]), init_pos[:, 0] - c[0]])


    stepper = StormerVerletStepper(particle_vom, dt, accel_expr)
    stepper.v.dat.data_wo[:] = V0 # assign initial velocity

    cell_crossing_solver = BisectionSolver()
    particle_traj_solver_params = ParticleTrajectorySolverParams(
        bary_tol=1e-9,
        abs_time_tol=1e-9,
        rel_time_tol=0,
        max_iters=50,
        plot=False
    )

    particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

    T_final = particle_traj_solver.solve(t_start, t_end)

    final_pos = particle_vom.coordinates.dat.data_ro

    surviving_particle_ids = particle_vom._particle_ids.dat.data_ro.copy()

    assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

    # Record final position error
    a = omega * t_end
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    exact_counterclockwise = c + (init_pos - c) @ R

    err = np.max(np.linalg.norm(final_pos - exact_counterclockwise, axis=1))
    errors.append(err)

# XXX: we use base log2 as the error halves at every level.
# For an arbitrary step sequence, use: p = log(e_k/e_{k+1})/log(dt_k/dt_{k+1})
rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
print("observed rates:", rates) # all rates are approx. 2 as expected

