from firedrake import *
import numpy as np

mesh = UnitSquareMesh(10, 10)

N = 10
particle_coords = np.random.rand(N, 2)
vom = VertexOnlyMesh(mesh, particle_coords)
io_vom = vom.input_ordering

print("Initial particle positions (in input order): ", particle_coords)
print("Initial particle positions (in primary VOM order): ", vom.coordinates.dat.data_ro)

FS_vom = VectorFunctionSpace(vom, "DG", 0, dim=2)
FS_io_vom = VectorFunctionSpace(io_vom, "DG", 0, dim=2)

fn = Function(FS_vom)
fn_io = Function(FS_io_vom)
vals = np.array([
    [10.5, 2.1],
    [2.4, 3.2],
    [5.6, 4.3],
    [7.8, 1.2],
    [0.5, 9.3],
    [4.4, 6.7],
    [8.8, 0.9],
    [3.3, 5.5],
    [1.1, 7.7],
    [9.9, 8.8]
])
fn_io.dat.data[:] = vals
fn.interpolate(fn_io)
print("Function values at particles (in VOM order): ", fn.dat.data_ro)

# Update the particle VOM by removing some particles.
# This causes the VOM topology to change so we need to reconstruct the DMswarm instead of mutating its fields.
particles_to_remove = [2, 5, 7] # assume indices in VOM ordering



