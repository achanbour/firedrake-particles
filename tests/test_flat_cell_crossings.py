from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
import particle_crossing_solver
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams


num_cells = [2, 5, 10, 15, 20, 30]
errors = []
bisection_calls = []

for n in num_cells:
    # Define the parent mesh
    parent_mesh = UnitSquareMesh(n, n, quadrilateral=False)

    # Define the particles VOM

    # Place a single particle close to a facet
    # For n = 2, a facet contains the point (0.625, 0.625)
    # For n = 10,  a facet contains the point (0.145, 0.145)
    # x0 = np.array([[0.140, 0.214]])
    
    # Place a single particle close to the bottom left corner (0, 0)
    x0 = np.array([[0.001, 0.001]])
    particle_vom = VertexOnlyMesh(parent_mesh, x0)
    x0_vom = particle_vom.coordinates.dat.data_ro.copy()

    # Define per-particle velocities
    V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
    v = Function(V, name="particle_velocity")
    v.dat.data[:] = [0.15, 0.156]

    # Define solvers

    # Make the particle travel at most half a cell width to ensure that no more than one crossing occurs
    # regardless where the particle started from
    dt = 0.5 * (1/n) / np.linalg.norm(v.dat.data_ro)
    t_start = 0
    t_end = dt # single time step
    stepper = ForwardEulerStepper(particle_vom, dt, v)

    cell_crossing_solver = particle_crossing_solver.BisectionSolver()
    particle_crossing_solver.BISECTION_COUNT = 0

    particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
    )
    particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

    T_final, surviving_particle_ids = particle_traj_solver.solve(t_start, t_end)
    
    print()
    print("Final particle position: ", particle_vom.coordinates.dat.data_ro)

    x_final_expected = x0_vom + T_final * v.dat.data_ro
    x_final_expected = x_final_expected[surviving_particle_ids]

    print("Expected final position: ", x_final_expected)
    err = np.linalg.norm(x_final_expected - particle_vom.coordinates.dat.data_ro)
    errors.append(err)

    bisection_calls.append(particle_crossing_solver.BISECTION_COUNT)


print("\nError convergence summary:")
print(f"{'n':>2} {'h':<8} {'crossings':>4} {'L2 error':>14}")
for n, cross, err in zip(num_cells, bisection_calls, errors):
    if cross > 1:
        continue
    print(f"{n:>2d} {1/n:>3f} {cross:>4d} {err:>20.6e}")

# plt.semilogy([n for n in num_cells], errors)
# plt.xlabel("Mesh resolution (num cells)")
# plt.ylabel("L2 error")
# plt.title("Crossing error in curved cells")
# plt.savefig("plots/crossing_error.png")
