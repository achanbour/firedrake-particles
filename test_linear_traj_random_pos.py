from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
from particle_traj_loop import move_particles_in_ref_space

"""
Deterministic particle trajectory test with random constant velocity (different across particles)
and random starting positions.
"""

np.random.seed(42)

t = 0.0 # current time
dt = 0.01 # time step
T = 1.0

# Define the parent mesh
mesh = UnitSquareMesh(10, 10, quadrilateral=False)
# mesh = PeriodicUnitSquareMesh(10, 10)

with PETSc.Log.Event("PreComputeCellFacetData"):
    _ = mesh.topology.cell_facet_neighbours
    _ = mesh.topology.cell_facet_coord_transforms

# Define the particles VOM
N = 10
particle_coords = np.random.rand(N, 2)
particle_vom = VertexOnlyMesh(mesh, particle_coords)
initial_particle_coords = particle_vom.coordinates.dat.data_ro.copy()
gdim = particle_vom.geometric_dimension
print("Initial particle positions (in primary VOM order): ", particle_vom.coordinates.dat.data_ro)

# Assign per-particle velocities
V = VectorFunctionSpace(particle_vom, "DG", 0, dim=gdim)
V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=gdim)
v = Function(V)
v_io = Function(V_io)
input_velocities = np.random.normal(0.1, 0.5, size=(N,2)) # 0.01
v_io.dat.data[:] = input_velocities
v.interpolate(v_io)
initial_particle_velocities = v.dat.data_ro.copy()

# Move particles in ref. space
# import timeit
with PETSc.Log.Event("ParticleTrajectoryLoop"):
    # t0 = timeit.default_timer()
    T_final, removed_particles = move_particles_in_ref_space(particle_vom, mesh, v, dt, T, t=0.0, plot=True)
    # t1 = timeit.default_timer()
    # print(f"[wall_time] {t1 - t0} s")
print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

from particle_traj_loop import BISECTION_COUNT
print("Number of bisection calls to resolve crossings: ", BISECTION_COUNT)

# from particle_time_stepper import STEP_COUNT
# print("Total ForwardEulerTimeStepper calls: ", STEP_COUNT)

# from pyop2.caching import print_cache_stats
# print_cache_stats()
# replace by: PYOP2_CACHE_INFO=1

# exact_final_coords_io = particle_coords + T_final*input_velocities 
# print("Exact final particle positions (IO): ", exact_final_coords_io)

exact_final_coords = initial_particle_coords + T_final*initial_particle_velocities
keep = np.ones(exact_final_coords.shape[0], dtype=bool)
keep[removed_particles] = False
exact_final_coords_survived = exact_final_coords[keep]

print("Exact final particle positions: ", exact_final_coords_survived)
print("Exact results matched up to 1e-5: ", np.allclose(particle_vom.coordinates.dat.data_ro, exact_final_coords_survived, atol=1e-5, rtol=0))
print("Exact results matched up to 1e-4: ", np.allclose(particle_vom.coordinates.dat.data_ro, exact_final_coords_survived, atol=1e-4, rtol=0))