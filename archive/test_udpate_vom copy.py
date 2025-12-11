from firedrake import *
import numpy as np

from update_vom import UpdateVertexOnlyMesh

np.random.seed(42)

# Define the parent mesh
mesh = UnitSquareMesh(10, 10)
x = SpatialCoordinate(mesh)

# Define the particles VOM
N = 10
particle_coords = np.random.rand(N, 2)
print("Initial particle coords: ",  particle_coords)

particle_vom = VertexOnlyMesh(mesh, particle_coords)
particle_vom_coord_fs = particle_vom.coordinates.function_space()

# Update particles positions
new_particle_coords = np.random.rand(N, 2)
print("New particle coords: ", new_particle_coords)

# Interpolate parent mesh into VOM
parent_mesh_interpolation = interpolate(x, particle_vom_coord_fs)

# Naive coord. update
particle_vom.coordinates.dat.data[:] = new_particle_coords
# print(assemble(parent_mesh_interpolation).dat.data)
# print("Parent mesh embedding diff (before update):", assemble(parent_mesh_interpolation).dat.data - particle_vom.coordinates.dat.data)

# Update the VOM
vom_updater = UpdateVertexOnlyMesh(particle_vom, mesh)
vom_updater.update(new_particle_coords)

# Interpolate after updating VOM
# parent_mesh_interpolation_after = assemble(interpolate(x, particle_vom_coord_fs))
# print("Particle coords (after update): ", particle_vom.coordinates.dat.data)
# print("Parent mesh embedding (after update): ", parent_mesh_interpolation_after.dat.data)
# print("Parent mesh embedding diff (after update):", assemble(parent_mesh_interpolation).dat.data - particle_vom.coordinates.dat.data)
print(assemble(parent_mesh_interpolation).dat.data)
