from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams
import particle_crossing_solver

"""
This experiment investigates the relation between the error of the numerical solution and the accuracy of the bisection algorithm
used to resolve cell crossings.

Using constant velocity, the numerical trajectory obtained using Forward Euler is exact at each time step.
Any error is therefore entirely attributable to bisection. We therefore want to make bisection as accurate as possible
(use absolute time tolerances only).
"""
# Parent mesh params
n_cells = 10

# Particle VOM params
n_particles = 10
x_diag = np.arange(n_particles) / 10.0 + 0.05
x0 = np.column_stack([x_diag, x_diag])
v0 = np.array([0.45, 0.25])

# Stepper params
t_start = 0
t_end = 1.5 # controls how much error gets accumulated
dt = 0.1 # controls how many crossings occur

bary_tol = 1e-16
abs_time_tols = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
rel_time_tol = 0

errors = []
bisection_calls = []

for abs_time_tol in abs_time_tols:
    # Rebuild a fresh mesh and a fresh particle VOM for each run
    parent_mesh = UnitSquareMesh(n_cells, n_cells, quadrilateral=False)
    particle_vom = VertexOnlyMesh(parent_mesh, x0)

    # Set the particles velocity
    V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
    v = Function(V, name="particle_velocity")
    V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=particle_vom.geometric_dimension)
    v_io = Function(V_io, name="io_particle_velocity")
    v_io.dat.data_wo[:] = v0
    v.interpolate(v_io)

    # Compute analytical solution
    x_analytical = particle_vom.coordinates.dat.data_ro + v.dat.data_ro * t_end
    keep = np.ones(x_analytical.shape[0], dtype=bool)
    
    # Set up timer stepper
    stepper = ForwardEulerStepper(particle_vom, dt, v)

    # Set up cell crossing solver
    # The number of bisection iterations needed to converge depends on time_tol
    # Since bisection halves the time bracket at each step, it converges once dt/2^n = time_tol
    max_bisection_iters = int(np.ceil(np.log2(dt/max(abs_time_tol, rel_time_tol * dt))))
    bisection_params = particle_crossing_solver.BisectionSolverParams(max_iters=max_bisection_iters)
    cell_crossing_solver = particle_crossing_solver.BisectionSolver(bisection_params)

    particle_crossing_solver.BISECTION_COUNT = 0

    # Set up the particle trajectory solver
    particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=bary_tol,
    abs_time_tol=abs_time_tol,
    rel_time_tol=0,
    max_iters=50,
    plot=False
    )

    particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)
    T_final, removed_particles = particle_traj_solver.solve(t_start, t_end)

    bisection_calls.append(particle_crossing_solver.BISECTION_COUNT)

    if len(removed_particles) > 0:
        keep[removed_particles] = False
        x_analytical = x_analytical[keep]
        
    x_numerical = particle_vom.coordinates.dat.data_ro
    err = np.linalg.norm(x_numerical-x_analytical)
    errors.append(err)

print("\nBisection convergence summary:")
print(f"{'abs_time_tol':>12} {'L2 error':>14} {'bisection_calls':>16}")
for tol, calls, err in zip(abs_time_tols, bisection_calls, errors):
    print(f"{tol:>12.0e} {err:>14.6e} {calls:>16d}")

# plt.semilogy(time_tols, errors)
plt.loglog(abs_time_tols, errors, label="L2 Error")
plt.xlabel("Absolute time tol")
plt.ylabel("L2 error")
plt.title("Bisection convergence")
plot_dir = "./plots"
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(f"{plot_dir}/bisection_error.png")

# NOTE:
# - When bisection converges, the final error is of the order of at least time_tol but it also depends on bary_tol: 
#   if bary_tol <= time_tol then the error is O(time_tol), 
#   if bary_tol > time_tol then the error is O(bary_tol)
#   so error = O(max(bary_tol, time_tol))
# - The number of bisection iterations is only relevant up to convergence (error is the same once bisection converged).
# - The error is expected to plateau once time_tol is small enough.
# - To hit the max. degree of accuracy assuming double precision, use abs_time_tol = 1e-16 and 
#   bary_tol=1e-16 (anything beyond that will likley be numerically unstable due to floating point noise).
