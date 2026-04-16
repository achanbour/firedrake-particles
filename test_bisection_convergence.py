from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

"""
This experiment investigates the dependence of the error in the numerical solution based on bisection accuracy.

Using constant velocity, the numerical trajectory obtained through the Forward Euler scheme is exact at each time step.
Any error is therefore entirely attributable to bisection, used to resolve cell crossings in each step.
"""
# Parent mesh
# Finer mesh implies smaller cells allowing for potentially more crossings
# E.g., n_cells=5: h=0.2 (coarse), n_cells=10: h=0.1 (fine)
n_cells = 10

# Initial particle positions
n_particles = 10
x_diag = np.arange(n_particles) / 10.0 + 0.05
x0 = np.column_stack([x_diag, x_diag])

# Velocity
# Speed controls how many crossings occur per time step
v0 = np.array([0.45, 0.25])

# Time parameters
# dt controls how many crossings occur (large: allow more crossings, small: few crossings)
# T controls how much crossing error gets accumulated (longer is better to accumulate signal?)
T = 1.5
dt = 0.1

# bisection_iters = [28, 30, 40, 50]
time_tols = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

errors = []
bisection_calls = []

for tol in time_tols:
    # Rebuild fresh mesh and particle VOM for each run
    mesh = UnitSquareMesh(n_cells, n_cells, quadrilateral=False)
    particle_vom = VertexOnlyMesh(mesh, x0)

    V = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
    v = Function(V, name="particle_velocity")
    V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=particle_vom.geometric_dimension)
    v_io = Function(V_io, name="io_particle_velocity")
    v_io.dat.data_wo[:] = v0
    v.interpolate(v_io)

    x_analytical = particle_vom.coordinates.dat.data_ro + v.dat.data_ro * T
    keep = np.ones(x_analytical.shape[0], dtype=bool)

    import particle_traj_loop as ptl
    ptl.BISECTION_COUNT = 0  # reset global counter

    # The number of bisection iterations needed to converge depends on time_tol
    # Since bisection halves the time bracket at each step, it converges once dt/2^n = time_tol
    max_iter = int(np.ceil(np.log2(dt/tol)))
    T_final, removed = ptl.solve_particle_traj_in_ref_space(
        particle_vom, mesh, v, dt, T, t=0.0,
        max_bisection_iters=max_iter,
        time_tol = tol,
        plot=False
    )
    
    bisection_calls.append(ptl.BISECTION_COUNT)

    if len(removed) > 0:
        keep[removed] = False
        x_analytical = x_analytical[keep]
        
    x_numerical = particle_vom.coordinates.dat.data_ro
    err = np.linalg.norm(x_numerical-x_analytical)
    errors.append(err)

print("\nBisection convergence summary:")
print(f"{'time_tol':>12} {'L2 error':>14} {'bisection_calls':>16}")
for tol, calls, err in zip(time_tols, bisection_calls, errors):
    print(f"{tol:>12.0e} {err:>14.6e} {calls:>16d}")

# plt.semilogy(time_tols, errors)
plt.loglog(time_tols, errors)
plt.xlabel("Time tol")
plt.ylabel("L2 error")
plt.title("Bisection convergence")
plt.savefig("plots/bisection_error.png")

# NOTE:
# - When the bisection algorithm does converge, its error is of the order of at most time_tol.
# - The number of bisection iterations is only relevant up to convergence (error is the same once bisection converged).
# - The error is expected to plateau once time_tol is small enough.