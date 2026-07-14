from firedrake import *
from pyop2.mpi import COMM_WORLD
from firedrake.petsc import PETSc


parent_mesh = UnitSquareMesh(5, 5)

# XXX: [[0.31, 0.99]] lands on rank 1
#      [[0.5, 0.5]] lands on rank 0
single_particle = [[0.31, 0.99]]

# XXX: With a single particle, `redundant=True | False` give the same outcome: the particle
# ends up in whichever rank owns its containing cell.
particle_vom = VertexOnlyMesh(parent_mesh, single_particle, redundant=True)

# PETSc.Sys.Print(f"\n[rank {COMM_WORLD.rank}]\n"
#                     f"  owns {particle_vom.cell_set.size} "
#                     f"(+ {particle_vom.num_vertices() - particle_vom.cell_set.size} halos)\n"
#                     f"  coords:\n{particle_vom.coordinates.dat.data_ro}\n",
#                     comm=PETSc.COMM_SELF
#                 )

# XXX: What does cell_facet_neighbours return?
n_owned = parent_mesh.cell_set.size
n_particles_owned = particle_vom.cell_set.size
if n_particles_owned > 0:
    parent_cell_num = particle_vom.topology.cell_parent_cell_list[0]
    print(f'rank {COMM_WORLD.rank} owns the particle in parent cell {parent_cell_num}')

    breakpoint()
    for i in range(3):
        next_cell = parent_mesh.topology.cell_facet_neighbours.data[parent_cell_num, i]
        if next_cell == -1:
            status = "exterior"
        elif next_cell < n_owned:
            status = "owned"
        else:
            status = "ghost"
        print(f"rank {COMM_WORLD.rank}: next cell across facet {i} is {next_cell} ({status})")
        breakpoint()