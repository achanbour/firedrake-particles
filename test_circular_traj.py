from firedrake import *
import numpy as np
from particle_traj_loop import move_particles_in_ref_space

"""
Deterministic particle trajectory test using a solid body rotation field with constant angular speed
and starting positions aligned on a circle centered at c with radius c.
"""

# Params
N = 10 # number of particles
r = 0.25 # radius
c = np.array([0.5, 0.5]) # center
theta = np.linspace(0, 2*np.pi, N, endpoint=False) # initial angles, evenly spaced on [0, 2pi)

# Initial positions
x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

radii = np.sqrt((q0[:, 0] - c[0])**2 + (q0[:, 1] - c[1])**2)
print(np.allclose(radii, r))

# Domain
mesh = UnitSquareMesh(10, 10, quadrilateral=False)

# VOM
vom = VertexOnlyMesh(mesh, q0)
print("Initial particle positions: ", vom.coordinates.dat.data_ro)

# Velocity field
# v(q) = omega * J * q -> linear in space so use CG1 FS
omega = 1.0 # angular speed
x = SpatialCoordinate(mesh)
v_expr = omega * as_vector([-x[1]+c[1], x[0]-c[0]])
V = VectorFunctionSpace(mesh, "CG", 1) 
v = Function(V, name="velocity_field")
v.interpolate(v_expr)

# Reconstruct the particle's trajectory
T = 1
dt = 0.01
T_final, removed_particles = move_particles_in_ref_space(vom, mesh, v, dt, T, t=0.0, plot=True)
print("Final particle positions: ", vom.coordinates.dat.data_ro)
print("Removed particles: ", removed_particles)

# NOTE: Forward Euler is not exact in this case.
# Visually, we can see particles moving away from the center of the circle they started on.
# This drift is due to the fact that, under Forward Euler, the radius ||q^n - c||^2 
# grows by a factor O(1+dt^2*omega*2) at each step.
# For more accurate results, use a structure-preserving scheme such as the implicit midpoint rule.








