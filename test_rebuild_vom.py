from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater

"""
Rebuild a VertexOnlyMesh following a topology change (e.g., removing some particles).
"""

mesh = UnitSquareMesh(10, 10)

N = 10
particle_coords = np.random.rand(N, 2)
vom = VertexOnlyMesh(mesh, particle_coords)
io_vom = vom.input_ordering

print("Initial particle positions (in input order): ", particle_coords)
print("Initial particle positions (in primary VOM order): ", vom.coordinates.dat.data_ro)
print("Initial particle reference positions: ", vom.reference_coordinates.dat.data_ro)

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
# print("Function values at particles (in VOM order): ", fn.dat.data_ro)

# Update the particle VOM by removing some particles.
# This causes the VOM topology to change so we need to reconstruct the DMswarm instead of mutating its fields.
particles_to_remove = [2, 5, 7] # assume indices in VOM ordering

vom_updater = VertexOnlyMeshUpdater(vom, mesh)
_, sf_old_to_new = vom_updater.rebuild_vom(particles_to_remove)

# NOTE: Changing the VOM topology invalidates the Function Spaces and Functions.
# The first step is to check that the VOM has been properly updated.

print("Updated particle positions: ", vom.coordinates.dat.data_ro)
print("Updated particle reference positions: ", vom.reference_coordinates.dat.data_ro)

# Next, we check the input ordering SF considering we've used the current VOM as IO VOM for the updated VOM.
# The IO SF maps vertices in the input ordering VOM (old) to vertices in the primary VOM (new).
print("Input ordering SF for updated VOM: ")
vom.input_ordering_sf.view()

# Inspect the SF we constructed that maps new VOM indices to old VOM indices.
# In this case it happens to be merely a compression of indices since the global order of particles was preserved.
# This is because we are running in serial (so no rank redistribution occurs).
print("Old-to-new VOM index SF: ")
print(sf_old_to_new.view())

# NOTE: Comparing the two SFs, it seems like both achieve the same mapping but in a different way.
# The input ordering SF maps old IO VOM indices to new primary VOM indices,
# while our constructed SF maps new VOM indices to old primary VOM indices.
# --> Q: Which one is the one to keep?

# print("Function values at particles after VOM update: ", fn.data.data_ro)

# TODO 1: Currently, the Functions are still tied to the old VOM so they return values using the old number of particles.
# We therefore need to:
# 1. redefine Functions or 
# 2. force them to use the new VOM topology based on the VOM version number.
#   This involves updating the FunctionSpaces' DM reference to the new VOM DMswarm,
#   possibly requiring to build an SF between the DM Sections defining the new and old FunctionSpaces.

# TODO 2: 
# - Inspect how the updater behaves when particle positions are updated in addition to some particles beind removed
#   which is currently the case in the particle tracking loop.
# - Determine what changes need to be made when redistribution occurs.