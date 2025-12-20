from firedrake import *
import numpy as np
from update_vom import UpdateVertexOnlyMesh

"""
Update a VertexOnlyMesh by changing particle positions in physical space.
"""

np.random.seed(42)

# Define the parent mesh
mesh = UnitSquareMesh(10, 10)
x = SpatialCoordinate(mesh)

# Define the particles VOM
N = 10
particle_coords = np.random.rand(N, 2)
print("Initial particle coords (in input order): ",  particle_coords)

particle_vom = VertexOnlyMesh(mesh, particle_coords)
particle_vom_coord_fs = particle_vom.coordinates.function_space()
print("Initial particle coords (in primary VOM order): ",  particle_vom.coordinates.dat.data)
# print("Particle coords. version: ", particle_vom.coordinates.dat.dat_version)

# Define a function space on the particles VOM
V = VectorFunctionSpace(particle_vom, "DG", 0, dim=2)
V_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=2)

v = Function(V)
v_io = Function(V_io)

# Update particles positions
new_particle_coords_input = np.random.rand(N, 2)
print("New particle coords (in input order): ", new_particle_coords_input)

v_io.dat.data[:] = new_particle_coords_input
v.interpolate(v_io)
print("New particle coords (in primary VOM order): ", v.dat.data_ro)

# Interpolation of parent mesh into VOM
parent_mesh_interpolation = interpolate(x, particle_vom_coord_fs)

# Naive coord. update
# particle_vom.coordinates.dat.data[:] = new_particle_coords
# print(assemble(parent_mesh_interpolation).dat.data)
# print("Parent mesh embedding diff (before update):", assemble(parent_mesh_interpolation).dat.data - particle_vom.coordinates.dat.data)

# Check nodes of FS before update
# print("Properties of the FS before mesh update:")
# node_coords = assemble(interpolate(particle_vom.coordinates, V))
# print("The coords. of the nodes of V before update: ", node_coords.dat.data_ro)
# These match exactly the vertices of particle_vom
# print(particle_vom.coordinates.dat.data_ro)

# Update the VOM
vom_updater = UpdateVertexOnlyMesh(particle_vom, mesh)
vom_updater.update(new_particle_coords_input) # NOTE: here we're using the new coordinates in input ordering!
# vom_updater.update(v.dat.data_ro) # NOTE: here we're using the new coordinates in primary VOM ordering

print("New particle coords (in updated VOM order): ", particle_vom.coordinates.dat.data)

# Interpolate after updating VOM
# parent_mesh_interpolation_after = assemble(interpolate(x, particle_vom_coord_fs))
# print("Parent mesh embedding (after update): ", parent_mesh_interpolation_after.dat.data)
# print("Parent mesh embedding diff (after update):", assemble(parent_mesh_interpolation).dat.data - particle_vom.coordinates.dat.data)
# print(assemble(parent_mesh_interpolation).dat.data)

# print("Particle coords. version: ", particle_vom.coordinates.dat.dat_version)
# NOTE: particle VOM coordinates version returns 4,
# meaning that dat is modified 4 times, how can I identify exactly where?

# Define a new function space and check equality with the previous function space
# V_new = VectorFunctionSpace(particle_vom, "DG", 0, dim=2)
# print(V == V_new)

# node_coords_updated = assemble(interpolate(particle_vom.coordinates, V))
# node_coords_new = assemble(interpolate(particle_vom.coordinates, V_new))
# print("Node coords. of V after update: ", node_coords_new.dat.data_ro)
# print("Node coords. of new V: ", node_coords_new.dat.data_ro)

# NOTE: The coords. of the FS DoFs/nodes match the coordinates of the vertices of the VOM
# as expected since DG0 on a VOM has defines one node per vertex. 



