import numpy as np
from firedrake import *
from ufl.differentiation import ReferenceGrad
from firedrake.interpolation import get_interpolator

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from update_vom import VertexOnlyMeshUpdater

parent_mesh = UnitSquareMesh(5,5)
x = SpatialCoordinate(parent_mesh)
invJ_expr = inv(ReferenceGrad(x))

coord = np.array([[0.5, 0.5]])
particle_vom = VertexOnlyMesh(parent_mesh, coord)
particle_vom_updater = VertexOnlyMeshUpdater(particle_vom, parent_mesh)

X = particle_vom.reference_coordinates

U = VectorFunctionSpace(particle_vom, "DG", 0)
u = Function(U)
u.dat.data_wo[:] = [0.15, 0.156]

TFS = TensorFunctionSpace(particle_vom, "DG", 0)

# Forward Euler expression
dt = 0.1
update_step = X + dt * interpolate(invJ_expr, TFS) * u
update_expr = interpolate(update_step, particle_vom.reference_coordinates.function_space()) # symbolic inteprolation expr

interpolator = get_interpolator(update_expr) # numerical interpolator
callable = interpolator._get_callable() # parloops

# res1 = assemble(update_expr)
res1 = callable()

# Mutate the VOM as in the inner loop: update ref. coords. + parent cell
current_parent_cells = partibcle_vom.topology.cell_parent_cell_list.copy()
next_parent_cells = current_parent_cells.copy()
next_cell = parent_mesh.topology.cell_facet_neighbours.data[current_parent_cells[0, 0], 0] # get neighbour across facet 0
next_parent_cells[0, 0] = next_cell
new_ref_pos = np.array([[0.5, 0.5]])  # same ref coords as initial

particle_vom_updater.update_ref_view(next_parent_cells, new_ref_pos)

# res2 = assemble(update_expr)
res2 = callable()


breakpoint()


# Calling assemble vs. executing parloops directly
#
# If we simply re-execute the parloop, we get res1 == res2 meaning the inner interpolate does not get re-evaluated.
# Its parloop was executed once at construction time during inside NestedInterpolateLowerer and its resulting Coefficient 
# was frozen into the outer parloop's args.
#
# If we call assemble then the inner interpolation expression seems to get re-evaluated. The reason is that assemble calls `get_interpolator(expr)`
# which reuses the cached Interpolator object. But then `interpolator.assemble(...)` is called which hits `self._get_callable` which calls `_build_interpolation_callables`
# which rebuilds the parloops and their arguments.