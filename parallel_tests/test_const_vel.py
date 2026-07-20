from firedrake import *
from pyop2.mpi import MPI

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_time_stepper_new import ForwardEulerStepper
from particle_crossing_solver import BisectionSolver
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

parent_mesh = UnitSquareMesh(10, 10, quadrilateral=False)
comm = parent_mesh.comm

# Vertical line of particles that move horizontally through the domain
num_particles = 10
y_line = np.linspace(0.15, 0.85, num_particles)
x0 = np.column_stack([np.full(num_particles, 0.25), y_line])

# XXX: redundant=True otherwise each rank must supply its own set of particles
particle_vom = VertexOnlyMesh(parent_mesh, x0)

# Constant velocity (same on all ranks)
# XXX: no need to go through IO VOM on initial assign
v0 = np.array([0.05, 0.0])
V = VectorFunctionSpace(particle_vom, "DG", 0)
v = Function(V, name="particle-velocity")
v.dat.data_wo[:] = v0

x0_local = particle_vom.coordinates.dat.data_ro.copy()
x0_global = np.concatenate(comm.allgather(x0_local))

t_start = 0.0
t_end = 10.0
dt = 0.1

stepper = ForwardEulerStepper(particle_vom, dt, v)
cell_crossing_solver = BisectionSolver()

particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False # XXX: does it run in parallel?
)
particle_traj_solver = ParticleTrajectorySolver(stepper, cell_crossing_solver, particle_traj_solver_params)

n_before = particle_traj_solver.num_particles_local
T_final = particle_traj_solver.solve(t_start, t_end)
n_after = particle_traj_solver.num_particles_local

surviving_particle_ids = particle_vom._particle_ids.dat.data_ro.copy()

# Check 1: did particles change ranks
print(f"[rank {comm.rank}] initially owned {n_before} particles, now owns {n_after} particles with pids {surviving_particle_ids}",
      flush=True)

# Check 2: no particle left the domain
n_survived = comm.allreduce(len(surviving_particle_ids), op=MPI.SUM)
assert n_survived == num_particles, f"lost particles: {n_survived} survived out of {num_particles}"

# Check 3: are the final positions correct?
# XXX: constant-velocity -> no discretisation error
x_expected = x0_global[surviving_particle_ids] + T_final*v0
err = np.linalg.norm(particle_vom.coordinates.dat.data_ro - x_expected, axis=1)
max_err = comm.allreduce(err.max() if len(err) else 0.0, op=MPI.MAX)

if comm.rank == 0:
    print(f"T_final={T_final}, global max position erorr = {max_err:.3e}")
    assert max_err < 1e-8, f"position error too large!"

