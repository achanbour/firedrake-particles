from firedrake import *
import particle_traj_loop_old as ptl
import numpy as np

num_cells = [2, 5, 10, 15, 20, 30]
# num_cells = [10]
errors = []
bisection_calls = []

for n in num_cells:
    mesh = UnitSquareMesh(n, n)

    # Place a single particle close to a facet
    # For n = 2, a facet contains the point (0.625, 0.625)
    # For n = 10,  a facet contains the point (0.145, 0.145)
    # x0 = np.array([[0.140, 0.214]])
    x0 = np.array([[0.001, 0.001]])
    particle_vom = VertexOnlyMesh(mesh, x0)
    x0_vom = particle_vom.coordinates.dat.data_ro.copy()

    # Assign per particle velocities
    U = VectorFunctionSpace(particle_vom, "DG", 0, dim=particle_vom.geometric_dimension)
    u = Function(U, name="particle_velocity")
    u.dat.data[:] = [0.15, 0.156]

    # Make the particle travel at most half a cell width to ensure that no more than one crossing occurs
    # regardless where the particle started from
    dt = 0.5 * (1/n) / np.linalg.norm(u.dat.data_ro)

    ptl.BISECTION_COUNT = 0 # reset global counter

    T_final, removed_particles = ptl.solve_particle_traj_in_ref_space(particle_vom, mesh, u, dt, dt, t=0.0, plot=False)
    
    print()
    print("Final particle position: ", particle_vom.coordinates.dat.data_ro)

    x_final_expected = x0_vom + T_final * u.dat.data_ro

    print("Expected final position: ", x_final_expected)
    err = np.linalg.norm(x_final_expected - particle_vom.coordinates.dat.data_ro)
    errors.append(err)

    # print("Total number of bisection calls: ", BISECTION_COUNT)
    bisection_calls.append(ptl.BISECTION_COUNT)


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
