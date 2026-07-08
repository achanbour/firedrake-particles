from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

from particle_time_stepper_new import StormerVerletStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

from vtk_output import write_pvd, write_particle_vtk

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

# Solver setup
t_start = 0
dt = 1e-3
t_end = 0.2 # 200 steps

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

VTKFile("output.pvd").write(accel_fn)
Path("particles").mkdir(exist_ok=True)
base = f"particles/particles_{{}}.vtu"

# Save initial frame
write_particle_vtk(base.format(0), particle_vom, stepper.v)

particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params, vtk_dir="particles", vtk_base=base)

T_final, surviving_particle_ids = particle_traj_solver.solve(t_start, t_end)

final_pos = particle_vom.coordinates.dat.data_ro

assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

print("Final particle positions:", final_pos)

# exact solution: uniform circular motion at angular rate omega
# same as in steady rigid-body rotation case
a = omega * t_end
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
exact_counterclockwise = c + (init_pos - c) @ R
# exact_clockwise = c + (init_pos - c) @ R.T

# Check numerical error: second order-accurate implies error in the final solution is O(dt^2)
# XXX: a more robust check involves halving the time step and checking that the error shrinks by a factor of 4.
print("max position error vs exact counterclockwise:", np.max(np.linalg.norm(final_pos - exact_counterclockwise, axis=1)))


# Check radius conservation proxy for energy conservation
# XXX: radius is not preserved exactly, but the drift remains bounded due to the symplecticity of the method
r0 = np.linalg.norm(init_pos - c, axis=1)
rT = np.linalg.norm(final_pos - c, axis=1)
radius_drift = np.max(np.abs(rT - r0))
print("max radius drift:", radius_drift)
assert radius_drift < 1e-2 * r, "radius not conserved)"

