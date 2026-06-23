from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper import ForwardEulerStepper
import particle_crossing_solver
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

num_cells = [2, 5, 10, 15, 20, 30]

errors_no_crossing = [] # Forward Euler error only - scales as O(dt^2)=O(h^2)
errors_with_crossing = [] # Forward Euler + any errors due to crossing (bisection, cell transition etc.)

bisection_calls_no_crossing = []
bisection_calls_with_crossing = []

for n in num_cells:
    # Define the parent mesh
    h = 1/n 
    parent_mesh = UnitSquareMesh(n, n)

    V = VectorFunctionSpace(parent_mesh, "CG", 2)
    v = Function(V)
    s, t = SpatialCoordinate(parent_mesh)

    # x_new = s
    # y_new = t + 0.2 * sin(pi*s)*sin(pi*t)

    x_new = s + 0.5 *s*(1-s)
    y_new = t + 0.5 *t*(1-t)

    # Define the parent curved mesh
    v.interpolate(
        as_vector([x_new, y_new])
    )
    curved_parent_mesh = Mesh(v)

    x0 = np.array([[0.001, 0.001]])
    u0 = np.array([0.15, 0.156])
    speed = np.linalg.norm(u0)

    # Make the particle travel at most half a cell width to ensure that no more than one crossing occurs
    # regardless where the particle started from
    dt = 0.5 * h / speed
    t_start = 0
    t_end = 1.5 * (h - 0.001) / speed

    # Can set dt to be tiny to shrink the Forward Euler error but run long enough so that a crossing occurs
    # dt_small = 0.01 * (1/n)**2 / np.linalg.norm(u.dat.data_ro)  # O(h^2)
    
    # Run the loop twice: once without crossing and once long enough so that a single crossing occurs
    for T, errors_list, bisection_calls_list in [
        (dt, errors_no_crossing, bisection_calls_no_crossing),
        (t_end, errors_with_crossing, bisection_calls_with_crossing)
    ]:
        
        particle_vom = VertexOnlyMesh(curved_parent_mesh, x0)
        x0_vom = particle_vom.coordinates.dat.data_ro.copy()

        # Assign per particle velocities
        U = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
        u = Function(U, name="particle_velocity")
        u.dat.data[:] = u0

        stepper = ForwardEulerStepper(particle_vom, dt, u)

        cell_crossing_solver = particle_crossing_solver.BisectionSolver()
        particle_crossing_solver.BISECTION_COUNT = 0

        particle_traj_solver_params = ParticleTrajectorySolverParams(
        bary_tol=1e-9,
        abs_time_tol=1e-9,
        rel_time_tol=1e-9,
        max_iters=50,
        plot=False
        )
        particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

        T_final, surviving_particle_ids = particle_traj_solver.solve(t_start, T)

        x_final_expected = x0_vom + T_final * u.dat.data_ro
        x_final_expected = x_final_expected[surviving_particle_ids]
        
        err = np.linalg.norm(x_final_expected - particle_vom.coordinates.dat.data_ro)
        errors_list.append(err)

        bisection_calls_list.append(particle_crossing_solver.BISECTION_COUNT)


# --- Check how error scales with h --
# print("\nError convergence summary:")
# print(f"{'n':>2} {'h':<8} {'crossings':<6} {'error':<14}")
# for n, cross, err in zip(num_cells, bisection_calls_with_crossing, errors_with_crossing):
#     print(f"{n:>2d} {1/n:.6f} {cross:>6d} {err:>14.6e}")

# plt.semilogy([n for n in num_cells], errors_with_crossing)
# plt.xlabel("Mesh resolution (num cells)")
# plt.ylabel("L2 error")
# plt.title("Crossing error in curved cells")
# plt.savefig("plots/crossing_error.png")


# --- Check Forward Euler error and cell transition error separately ---
# Compute cell transition error by subtracting the Forward Euler error
errors_transition = [e_cross - e_no_cross for e_cross, e_no_cross in zip(errors_with_crossing, errors_no_crossing)]

print("\nError convergence summary:")
print(f"{'n':>2} {'h':<8} {'no-crossing':>14} {'O(h^2)?':>14} {'with-crossing':>14} {'cell transition':>14} {'O(h^2)?':>14}")
for n, e_no, e_cross, e_trans in zip(num_cells, errors_no_crossing, errors_with_crossing, errors_transition):
    print(f"{n:>2d} {1/n:.6f} {e_no:>14.6e} {e_no/((1/n)**2):>14.6e} {e_cross:>14.6e} {e_trans:>14.6e} {e_trans/((1/n)**2):>14.6e}")

# --- NOTE: Experimental setup ---
# > For the case with 0 crossings: use T=dt (single time step) and dt=O(h) so that the Forward Euler error is O(dt^2)=O(h^2)
# Note that error must equal C*h^2 for some problem dependent constant C
# To verify, divide each error (for each h) by h^2 and check that the constant C is the same
# > For the case with 1 crossing: use tiny dt to shrink the Forward Euler error and measure the error in the affine approx. of cell transition
# facet coord transforms. Use the same logic as above to verify that the error is O(h^2).