from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

"""
This experiment focusses on the bisection error in the particle trajectory loop.

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
# ---
# At T = 2.0, only one particle remains
# At T = 2.4, the remaining particle leaves the domain so new VOM is empty
# ---
T = 1.5 # 1.0, 1.5, 2.0
dt = 0.1

bisection_iters = [1, 2, 4, 8, 16, 20, 30] # 20
errors = []
bisection_calls = []

for iter in bisection_iters:
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

    T_final, removed = ptl.move_particles_in_ref_space(
        particle_vom, mesh, v, dt, T, t=0.0,
        max_bisection_iters=iter,
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
print("Bisection calls: ", bisection_calls)
print(f"{'max_iters':>12} {'L2 error':>14}")
for iters, call, err in zip(bisection_iters, bisection_calls, errors):
    print(f"{iters:>12d} {err:>14.6e}")

plt.semilogy(bisection_iters, errors)
plt.xlabel("Number of bisection iterations")
plt.ylabel("L2 error")
plt.title("Bisection convergence")
plt.savefig("plots/bisection_error.png")

# NOTE: bisection error findings:
# - The error decreases with the number of bisection iterations and saturates at `time_tol`









