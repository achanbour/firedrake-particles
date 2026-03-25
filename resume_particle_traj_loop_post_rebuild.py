from firedrake import *
import pickle
import numpy as np
from particle_traj_loop import move_particles_in_ref_space
from update_vom import VertexOnlyMeshUpdater

# Pickle file stores all the field dats after the VOM has been rebuilt
with open("particle_loop_dats.pickle", "rb") as file:
    prev_iter_dats = pickle.load(file)

particle_pos = prev_iter_dats["x"]
particle_ref_pos = prev_iter_dats["x_ref"]
invJ = prev_iter_dats["invJ"]
particle_velocity =  prev_iter_dats["v"]
particle_dt =  prev_iter_dats["dt"]

mesh = UnitSquareMesh(10, 10, quadrilateral=False)
N = 9
particle_coords = np.random.rand(N, 2) # dummy coords. to initialize VOM
particle_vom = VertexOnlyMesh(mesh, particle_coords)

# Store the actual particle coords. in a temporary Function
V = VectorFunctionSpace(particle_vom, "DG", 0)
tmp_coord_func = Function(V)
tmp_coord_func.dat.data_wo[:] = particle_pos

# Update VOM coords. through VOM updater
vom_updater = VertexOnlyMeshUpdater(particle_vom, mesh)
vom_updater.update(tmp_coord_func)

assert np.allclose(particle_vom.coordinates.dat.data_ro, particle_pos)
assert np.allclose(particle_vom.reference_coordinates.dat.data_ro, particle_ref_pos)

velocity_func = Function(V)
velocity_func.dat.data_wo[:] = particle_velocity

print("Initial particle positions: ", particle_vom.coordinates.dat.data_ro)
T_final, removed_particles = move_particles_in_ref_space(particle_vom, mesh, v_fn=velocity_func, t=0.04, dt=0.01, T=0.05)
print("Removed particles: ", removed_particles)
print("Final particle positions: ", particle_vom.coordinates.dat.data_ro)

"""
Exact final particle positions:  [[ 0.19315986  0.15085011]
 [ 0.1672823   0.23021146]
 [ 0.45300864  0.26120805]
 [ 0.30440481  0.49881366]
 [ 0.06027182  0.83105744]
 [-0.00769035  0.9798023 ]
 [ 0.81792667  0.20554677]
 [ 0.70979334  0.56385089]
 [ 0.34971934  0.95907049]
 [ 0.58800544  0.71134564]]
"""