from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater
import sys
import gc

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

FS_vom = VectorFunctionSpace(vom, "DG", 0, dim=2) # a FS on the primary VOM
FS_io_vom = VectorFunctionSpace(io_vom, "DG", 0, dim=2) # a FS on the input ordering VOM

fn = Function(FS_vom) # a Function on the primary VOM
fn_io = Function(FS_io_vom) # a Function on the input ordering VOM
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

# Update the particle VOM by removing some particles
# This causes the VOM topology to change so we need to reconstruct a new DMswarm on the updated particle set.
particles_to_remove = [2, 5, 7] # assume indices to be in VOM ordering

vom_updater = VertexOnlyMeshUpdater(vom, mesh)
vom_updater.rebuild_vom(particles_to_remove)

# The first step is to check that the VOM has been properly updated.
print("Updated particle positions: ", vom.coordinates.dat.data_ro)
print("Updated particle reference positions: ", vom.reference_coordinates.dat.data_ro)
print("VOM version: ", vom._topology_version)

## ---
# Check the input ordering SF considering we've used VOM_0 as IO VOM when rebuilding VOM_0 to VOM_1.
# The IO SF maps vertices in the input ordering VOM (old) to vertices in the primary VOM (new).
# print("Input ordering SF for updated VOM: ")
# vom.input_ordering_sf.view()

# Note that, in this case, the global order of particles is preserved. This happens because:
# - we embedd the old coordinates of VOM_0 and create a new swarm picking out only surviving particles
# - no redistribution occurs as we're running in serial.

## ---
# Changing the VOM topology invalidates the Function Spaces and Functions.
# We attempt to rebuild the Function Spaces and Functions on the updated VOM internally
# when they are next accessed.
print("Function values at particles after VOM update: ", fn.dat.data_ro)

# print(fn.function_space() is FS_vom)
#print(fn.function_space().cell_node_list)

# The above returns False as the FS reference has been updated to a new FS which is now defined on the updated VOM.
# In rebuild_vom, we have changed the ._topology object. Given that, FS cache keys involve the mesh topology, a new topology means a new cache owner 
# so we don't hit stale caches when accessing the new FS data.

# Check references to the old FS
# `getrefcount` returns the total number of references to an object (including temporary references and multiple refs from the same container).
# `get_referrers` returns a list of distinct objects referencing the given object.
# print("References to old FS after rebuilding VOM + Function: ",  sys.getrefcount(FS_vom))
# referrers = gc.get_referrers(FS_vom)
# print(f"Number of referrers: {len(referrers)}")
# for i in range(len(referrers)):
#     print(f"Referrer {i}: {referrers[i]}")
# print(any(ref is fn for ref in referrers)) # returns False i.e., fn no longer holds a reference to the old FS

## ---
# TODO 1: 
# Currently, the Functions are still tied to the old VOM so they return values using the old number of particles.
# We therefore need to:
# 1. redefine Functions or 
# 2. force them to use the new VOM topology based on the VOM version number.
#   This involves updating the FunctionSpaces DM reference to the new VOM DMswarm,
#   possibly requiring to build an SF between the DM Sections defining the new and old FunctionSpaces.

# TODO 2: 
# - Inspect how the updater behaves when particle positions are updated in addition to some particles being removed
#   which is currently the case in the particle tracking loop.
# - Determine what changes need to be made when redistribution of particles occurs.