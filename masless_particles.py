from firedrake import *
import numpy as np

from update_vom import RebuiltVertexOnlyMesh

# Instead of solving the Lagrangian ODE dx/dt = v where each particle carries its own velocity,
# we solve dx/dt = v where v is a velocity field defined over the parent mesh (v depends on x and possibly on t)

# Define a time-integration scheme to move particles
t = 0.0
dt = 0.1
T = 1.0

def move_particles(pmesh, t, dt, T, v, v_expr=None):
    """
    Solve dx/dt = v(x,t) by forward Euler.

    v_expr is a UFL expression which may depend on t.
    v is the Function into which we interpolate v_expr each step.
    """
    while t < T:
        if v_expr is not None:
            # update the velocity field at time t
            v.interpolate(v_expr)

        # evaluate velocity at particle positions
        pcoords = pmesh.coordinates              
        v_at_particles = assemble(interpolate(v, pcoords.function_space()))

        # forward euler update
        new_coords = assemble(interpolate(pcoords + dt * v_at_particles, pcoords.function_space()))

        # update (rebuild) the particle meshs
        pmesh.update(new_coords.dat.data_ro)

        t += dt
    return pmesh


# Define the parent mesh
mesh = UnitSquareMesh(2, 2)
x, y = SpatialCoordinate(mesh)

# Define the velocity field
V = VectorFunctionSpace(mesh, "CG", 1)
v = Function(V)

# Define velocity
# v_expr = as_vector([x*(1-x), y*(1-y)])
v_expr = as_vector([sqrt(t) + x*(1-x), sqrt(t) + y*(1-y)])
v.interpolate(v_expr)

# Define the particles in a VOM
N = 1
particle_coords = np.random.rand(N, 2)
particle_vom = VertexOnlyMesh(mesh, particle_coords)
print("Initial particle positions (particle VOM): ", particle_vom.coordinates.dat.data_ro)

# Wrap the particles VOM in an "updater" (DynamicVertexOnlyMesh OR RebuiltVertexOnlyMesh)
# that rebuilds the VOM every time particle positions change
particle_vom_updater = RebuiltVertexOnlyMesh(particle_vom, mesh)

particles_updated = move_particles(particle_vom_updater, t, dt, T, v, v_expr)

print("Final particle positions (updated particle VOM): ", particles_updated.coordinates.dat.data_ro)

embedding_func = assemble(interpolate(SpatialCoordinate(mesh), particles_updated.coordinates.function_space()))
print(embedding_func.dat.data_ro - particles_updated.coordinates.dat.data_ro)



