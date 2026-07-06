from firedrake import *
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_crossing_solver import BisectionSolver
from particle_time_stepper_new import ForwardEulerStepper
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

from vtk_output import write_particle_vtk, write_pvd

mesh = UnitSquareMesh(40, 40, quadrilateral=False)
V = VectorFunctionSpace(mesh, "CG", 2) # velocity FS

num_particles = 10
r = 0.25
c = np.array([0.5, 0.5])
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)
q0 = np.column_stack([c[0] + r*np.cos(theta), c[1] + r*np.sin(theta)])

particle_vom = VertexOnlyMesh(mesh, q0)
P0DG = VectorFunctionSpace(particle_vom, "DG", 0)

# Define the Taylor-Green vortex (steady)
U0 = 40.0 # fluid speed -> determines how many cell crossings occur per time step
x, y= SpatialCoordinate(mesh)
u_expr = U0 * as_vector([sin(pi*x)*cos(pi*y),
                         -cos(pi*x)*sin(pi*y)])

# Store the velocity field in a Function to plot streamlines in Paraview
# (curves tangent to the velocity field everywhere)
V = VectorFunctionSpace(mesh, "CG", 2)
u_fn = Function(V, name="Velocity")
u_fn.interpolate(u_expr)

# --- Solvers ---
dt, timesteps = 1e-3, 200
# stepper = ForwardEulerStepper(particle_vom, dt, u_fn)
stepper = ForwardEulerStepper(particle_vom, dt, u_expr)
params = ParticleTrajectorySolverParams(bary_tol=1e-9, abs_time_tol=1e-9,
                                        rel_time_tol=0, max_iters=50, plot=False)
solver = ParticleTrajectorySolver(stepper, BisectionSolver(), params)

# ---- output ----
VTKFile("output.pvd").write(u_fn)
Path("particles").mkdir(exist_ok=True)
base = "particles/particles_{}.vtu"
count = 0

# Evaluate the velocity field at the particles
u_at_points = Function(P0DG)
u_at_points.interpolate(u_fn)

time = 0.0
for timestep in range(timesteps):
    write_particle_vtk(base.format(count), particle_vom, u_at_points)
    count += 1

    time_start = time
    stepper._dt = dt
    time += dt
    # u_fn.interpolate(u_expr)
    solver.solve(time_start, time)
    u_at_points.interpolate(u_fn)

write_pvd("particles.pvd", base, count)