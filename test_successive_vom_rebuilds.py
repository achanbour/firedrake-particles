from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater

mesh = UnitSquareMesh(10, 10)

N = 10
particle_coords = np.random.rand(N, 2)
vom = VertexOnlyMesh(mesh, particle_coords)
io_vom = vom.input_ordering

# print("Initial particle positions (in input order): ", particle_coords)
# print("Initial particle positions (in primary VOM order): ", vom.coordinates.dat.data_ro)
# print("Initial particle reference positions: ", vom.reference_coordinates.dat.data_ro)

FS_vom = VectorFunctionSpace(vom, "DG", 0, dim=2)
FS_io_vom = VectorFunctionSpace(io_vom, "DG", 0, dim=2) 

fn_0 = Function(FS_vom)
fn_1 = Function(FS_vom)

fn_io= Function(FS_io_vom)

vals_0 = np.array([
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

vals_1 = np.array([
    [6.4, 5.8],
    [3.6, 0.4],
    [1.9, 3.4],
    [8.7, 5.5],
    [0.2, 5.7],
    [3.3, 0.1],
    [4.1, 2.6],
    [6.1, 7.5],
    [9.7, 8.3],
    [1.2, 7.3]
])

fn_io.dat.data[:] = vals_0
fn_0.interpolate(fn_io)
print("Function 0 values on VOM_0: ", fn_0.dat.data_ro)

fn_io.dat.data[:] = vals_1
fn_1.interpolate(fn_io)
print("Function 1 values on VOM_0: ", fn_1.dat.data_ro)


vom_updater = VertexOnlyMeshUpdater(vom, mesh)

breakpoint()

# VOM rebuild 1 
particles_to_remove = [2, 5, 7]
new_coords = vom.coordinates.dat.data_ro - 0.01 * np.array([1.0, 0.0]) # change x coordinates only
vom_updater.rebuild_vom(particles_to_remove, new_coords)
# Access Function 0 (Function rebuild uses SF_{1,0})
print("Function 0 values on VOM_1: ", fn_0.dat.data_ro)

breakpoint()

# VOM rebuild 2
particles_to_remove = [0] 
new_coords = vom.coordinates.dat.data_ro - 0.01 * np.array([0.0, 1.0]) # change y coordinates only
vom_updater.rebuild_vom(particles_to_remove, new_coords)
# Access Function 1 (Function rebuild uses SF_{2,1} o SF_{1,0} -> SF_{2,0})
print("Function 1 values on VOM_2: ", fn_1.dat.data_ro)