from firedrake import *
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from particle_time_stepper import ForwardEulerStepper
from particle_time_stepper_new import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

"""
Circular particle trajectory using a rigid-body rotation field with constant angular speed.
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

# Define the velocity field (lives on the parent mesh )
# v(q) = omega * J * q
# omega = 0.5 # angular speed
omega = 10*np.pi
x = SpatialCoordinate(parent_mesh)
v_expr = omega * as_vector([-x[1]+c[1], x[0]-c[0]])

V = VectorFunctionSpace(parent_mesh, "CG", 1)
v_fn = Function(V, name="velocity").interpolate(v_expr)

# ---- output ----
VTKFile("output.pvd").write(v_fn)
particle_file = VTKFile("particles.pvd")

# Define the solvers
t_start = 0
# t_end = 1
# dt = 0.01
t_end = 0.2 # equal to the period T with omega = 10*pi
dt = 1e-3
stepper = ForwardEulerStepper(particle_vom, dt, v_expr)

cell_crossing_solver = BisectionSolver()

particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)

r_hist, t_hist = [], []

def radius_monitor(t, solver):
    pos = solver.particle_vom.coordinates.dat.data_ro
    r_hist.append(np.max(np.linalg.norm(pos - c, axis=1)))
    t_hist.append(t)
    particle_file.write(solver.stepper.v, time=t)

particle_traj_solver = ParticleTrajectorySolver(
    stepper, 
    cell_crossing_solver, 
    particle_traj_solver_params,
    monitor=radius_monitor
    )

# Initial coordinates in VOM order (before solve)
init_pos = particle_vom.coordinates.dat.data_ro.copy()

# Record initial frame
radius_monitor(t_start, particle_traj_solver)


T_final = particle_traj_solver.solve(t_start, t_end)

# Surviving particles (after solve)
surviving_particle_ids = particle_vom.firedrake_particle_ids.dat.data_ro.copy()

# Final coordinates (after solve)
# VOM does not get resized/redistributed so the ordering is preserved
final_pos = particle_vom.coordinates.dat.data_ro

assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

print("Final particle positions:", final_pos)

from particle_crossing_solver import BISECTION_COUNT
print("Number of bisection calls to resolve cell crossings: ", BISECTION_COUNT)

# NOTE: Forward Euler is not exact in this case.
# Visually, we can see particles moving away from the center of the circle they started on.
# This drift is due to the fact that, under Forward Euler, the radius ||q^n - c||^2 
# grows by a factor O(1+dt^2*omega*2) at each step.
# In reality, we are overestimating the FE error and the drift is smaller because of the crossing solver algorithm.

# For more accurate results, use a structure-preserving scheme such as the implicit midpoint rule.

# --- exact solution ---
a = omega * t_end
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
exact = c + (init_pos - c) @ R.T

# Check 1: radius matches the expected FE divergence
N = round((t_end - t_start) / dt)
r_pred = r * (1 + dt**2 * omega**2) ** (N / 2) # predicted radius after n steps
r_num  = np.linalg.norm(final_pos - c, axis=1) # actual radius
# assert np.allclose(r_num, r_pred, rtol=1e-3), (r_num, r_pred)

import matplotlib.pyplot as plt
r_arr = np.array(r_hist)
t_arr = np.array(t_hist)
plt.plot(t_arr, (r_arr-r_arr[0])/r_arr[0], label="forward Euler")
n = np.round(t_arr/dt)
plt.plot(t_arr, (1 + dt**2 * omega**2)**(n/2) - 1, "--", label="predicted")
plt.xlabel("t")
plt.ylabel("(r-r0)/r0")
plt.legend()
plt.savefig("fe_radius_drift", dpi=300)

# Check 2: angle advanced by exactly omega*t_end
ang0 = np.arctan2(init_pos[:,1]-c[1], init_pos[:,0]-c[0])
angN = np.arctan2(final_pos[:,1]-c[1], final_pos[:,0]-c[0])
dang = np.mod(angN - ang0, 2*np.pi)
# assert np.allclose(dang, omega*t_end, atol=1e-6), (dang, omega*t_end)

# Check 3: numerical error is O(dt)
print("max final position error:", np.max(np.linalg.norm(final_pos - exact, axis=1)))



