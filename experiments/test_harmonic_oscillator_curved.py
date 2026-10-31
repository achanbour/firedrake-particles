from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

from particle_time_stepper import StormerVerletStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

from output_paths import experiment_results_dir
from sv_stepper import accel, stormer_verlet_stepper

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

# initial positions starting on a circle centered at c with radius r
x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

# -- Define the geometry
parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)
V = VectorFunctionSpace(parent_mesh, "CG", 3)
v = Function(V)
x, y = SpatialCoordinate(parent_mesh)

# x_new = x + 0.5*x**2*(1-x)
# y_new = y + 0.5*y**2*(1-y)

x_new = x + 0.5*y*(1-y)
y_new = y + 0.5*x*(1-x)

v.interpolate(as_vector([x_new, y_new]))
curved_parent_mesh = Mesh(v)

particle_vom = VertexOnlyMesh(curved_parent_mesh, q0)

# -- Define the potential field
# Harmonic potential is V(q) = 0.5 * omega**2 * (x-c)**2
# The acceleration a is defined as a = -V'(q)
omega = 10*np.pi
x = SpatialCoordinate(curved_parent_mesh)
# accel_expr = -omega**2*as_vector([x[0]-c[0], x[1]-c[1]])
displacement = as_vector([x[0] - c[0], x[1] - c[1]])
potential_expr = 0.5*omega**2*inner(displacement, displacement)
accel_expr = -grad(potential_expr)

Va = VectorFunctionSpace(curved_parent_mesh, "CG", 1)
accel_fn = Function(Va, name="acceleration").interpolate(accel_expr)

# -- Initial velocity
init_pos = particle_vom.coordinates.dat.data_ro.copy()
# eccentricitiy 0.5 means 50% slower than a perfect circular motion -> ellipsis
V0 = 0.5 * omega * np.column_stack([-(init_pos[:, 1] - c[1]), init_pos[:, 0] - c[0]])

# Solver setup
t_start = 0
t_end = 1
dt = 1e-3

correction=True
tag = "with_correction" if correction else "without_correction"
stepper = StormerVerletStepper(particle_vom, dt, accel_expr, geometric_correction=correction)
stepper.v.dat.data_wo[:] = V0 # assign initial velocity

cell_crossing_solver = BisectionSolver()
particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)

# ---- output ----
output_dir = experiment_results_dir(__file__)
# VTKFile(str(output_dir / "output.pvd")).write(accel_fn)
# particle_file = VTKFile(str(output_dir / "particles.pvd"))


def total_particle_energy(pos, vel):
    """Total energy summed over all particles"""
    return np.sum(
        0.5*np.sum(vel ** 2, axis=1)
        + 0.5*omega**2*np.sum((pos - c)**2, axis=1)
    )

E_hist, t_hist = [], []
pos_hist, vel_hist = [], []
def energy_monitor(t, solver):
    pos = solver.particle_vom.coordinates.dat.data_ro
    vel = solver.stepper.v.dat.data_ro
    E_hist.append(total_particle_energy(pos, vel))
    t_hist.append(t)
    # particle_file.write(solver.stepper.v, time=t)
    # record full trajectory
    pos_hist.append(pos.copy())
    vel_hist.append(vel.copy())

particle_traj_solver = ParticleTrajectorySolver(
    stepper, 
    cell_crossing_solver, 
    particle_traj_solver_params,
    monitor=energy_monitor)


energy_monitor(t_start, particle_traj_solver) # record initial energy

T_final = particle_traj_solver.solve(t_start, t_end)

final_pos = particle_vom.coordinates.dat.data_ro

surviving_particle_ids = particle_vom._particle_ids.dat.data_ro

assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

print("Final particle positions:", final_pos)

# exact solution

# -- uniform circular motion at angular rate omega (same as in steady rigid-body rotation case)
# a = omega * t_end
# R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
# exact = c + (init_pos - c) @ R
# exact_clockwise = c + (init_pos - c) @ R.T

# -- general solution
exact = c + np.cos(omega*t_end) * (init_pos - c) + np.sin(omega*t_end) / omega * V0

# Check numerical error: second order-accurate implies error in the final solution is O(dt^2)
# XXX: a more robust check involves halving the time step and checking that the error shrinks by a factor of 4.
print("final max position error:", np.max(np.linalg.norm(final_pos - exact, axis=1)))


# Check radius conservation proxy for energy conservation
# XXX: radius is not preserved exactly, but the drift remains bounded due to the symplecticity of the method
r0 = np.linalg.norm(init_pos - c, axis=1)
rT = np.linalg.norm(final_pos - c, axis=1)
radius_drift = np.max(np.abs(rT - r0))
print("max radius drift:", radius_drift)
# assert radius_drift < 1e-2 * r, "radius not conserved"

# Check that the total energy of the system remains bounded throughout the simulation
import matplotlib.pyplot as plt
E = np.array(E_hist)
t = np.array(t_hist)
plt.plot(t, (E-E[0])/E[0])
plt.xlabel("t")
plt.ylabel("(E-E0)/E0")
plt.savefig(output_dir / f"curved_sv_energy_drift_{tag}.png", dpi=300)

print("max (E - E[0]))/E[0]:", np.max(np.abs(E - E[0]))/E[0])
# assert np.max(np.abs(E - E[0]))/E[0] < 1e-3, "energy not bounded"

# Compare Firedrake vs plain numpy SV
def particle_energy(x, v):
    """Energy of each particle"""
    return 0.5*np.sum(v**2, axis=-1) + 0.5*omega**2*np.sum((x-c)**2,axis=-1)

x_np, v_np = stormer_verlet_stepper(init_pos, V0, t, lambda x: accel(x, c, omega))
E_np = particle_energy(x_np, v_np)

x_fd, v_fd = np.array(pos_hist), np.array(vel_hist)

diff_x = np.linalg.norm(x_fd - x_np, axis=-1)
diff_E_rel = (particle_energy(x_fd, v_fd) - particle_energy(x_np, v_np)) / particle_energy(x_np, v_np)

# print("max |x_fd- x_np| at t_end:", diff_x[-1].max())
# print("max |E_fd - E_np|/E0 at t_end:", np.abs(diff_E_rel[-1]).max())

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].semilogy(t[1:], diff_x[1:].max(axis=1))
axes[0].set_ylabel("max |x_fd - x_np|")
axes[1].plot(t, diff_E_rel.max(axis=1), label="max over particles")
axes[1].plot(t, diff_E_rel.min(axis=1), label="min over particles")
axes[1].legend(loc="lower right")
axes[1].set_ylabel("|E_fd - E_np|/E0"); axes[1].set_xlabel("t"); axes[1].legend()
fig.savefig(output_dir / f"fd_vs_numpy_{tag}.png", dpi=300)






