from firedrake import *
import numpy as np

"""
This simple example demonstrates how a VertexOnlyMesh gets renumbered differently by Firedrake across independent runs.
"""

mesh = UnitSquareMesh(10, 10)
N = 10
point_coords = np.random.rand(N, 2)

vom = VertexOnlyMesh(mesh, point_coords)

# Inspect the permutation IS of the VOM.
# This gives a re-ordering of the DMSwarm points based on the parent mesh cell numbering
# `_renumbering_entities` gets the parent mesh DMPlex renumbering (old numbering -> new numbering), the parent cell IDs,
# and gets the inverse of the parent mesh renumbering (new numbering -> old numbering). 
# It then sorts particles by their original DMPlex cell numbers.
# This gives particles in original cell 0 first, then cell 1, then cell 2 ...

# The reason for undoing the parent mesh renumbering is because it's inherently non-deterministic 
# (due to hash randomization, MPI rank distribution etc.)
# but we want a deterministic ordering of the VOM.
# mesh partitioning (which rank owns which cells) and entity processing (marking entities as owned/halo)
# involve hash-based data structures (sets/dicts) where iteration order is non deterministic
# different hash values -> different position in internal arrays -> different iteration order
# -> different processing order -> different final entity numbering

vom_perm_is = vom.topology._dm_renumbering
parent_mesh_perm_is = vom._parent_mesh._dm_renumbering
breakpoint()
print(vom_perm_is.getIndices())
print(parent_mesh_perm_is.getIndices())

io_vom = vom.input_ordering
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