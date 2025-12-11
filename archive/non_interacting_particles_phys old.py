from firedrake import *
import numpy as np

from update_vom import RebuildVertexOnlyMesh, UpdateVertexOnlyMesh

np.random.seed(42)

# Define a time-integration scheme to move particles
t = 0.0
dt = 0.1
T = 1 

def move_particles(pmesh, v, dt, T, t=0.0):
    """
    Solve dx/dt = v by forward Euler.

    `pmesh` is assumed to be a Firedrake VertexOnlyMesh.
    """

    while t < T:
        coords = pmesh.coordinates # a Firedrake Function storing the physical coordinates of the particles 
        update_expr = coords + dt * v  # Firedrake UFL expression for the coordinates update 
        # new_coords = coords_func.interpolate(update_expr)   # a Firedrake Function storing the updated coordinates
        new_coords = assemble(interpolate(update_expr, coords.function_space()))
        t += dt
    return pmesh 

def move_particles_and_rebuild_vom(pmesh, v, dt, T, t=0.0):
    """
    Same as `move_particles` but rebuilds the particles VOM at each coord. change.
    
    `pmesh` is assumed to be `RebuildVertexOnlyMesh` object. This class has an internal method that
    provides attribute access to the underlying Firedrake VertexOnlyMesh object.

    Returns the final VOM.
    """
    while t < T:
        coords = pmesh.coordinates 
        
        # v must live on the current VOM
        assert v.function_space().mesh() is pmesh.vom

        update_expr = coords + dt * v                 
        new_coords = assemble(interpolate(update_expr, coords.function_space()))  

        # rebuild the VOM
        pmesh.update(new_coords.dat.data_ro)

        # assert v.function_space().mesh() is pmesh.vom
        
        # The above fails since FS are bound to specific meshes
        # as soon as a mesh is destroyed and recreated, its corresponding FS
        # must be redefined.

        V_new = VectorFunctionSpace(pmesh.vom, "DG", 0, dim=2)
        v_new = Function(V_new)
        v_new.dat.data[:] = v.dat.data[:]
        v = v_new

        # We get two different input ordering swarms for the VOM before and after reconstruction so interpolation fails
        # the only way is to define a new FS and function for velocity as above.

        # v = assemble(interpolate(v, V_new)) # ValueError: The target vom and source vom must be linked by input ordering!
        # print(pmesh.vom.input_ordering)
        # print(v.function_space().mesh().input_ordering)

        t += dt
    return pmesh.vom


def move_particles_and_update_vom(pmesh, mesh, v, dt, T, t=0.0):
    """
    Same as `move_particles_and_rebuild_vom` but updates the VOM rather than rebuilds a new one at every coord. change.
    
    `pmesh` is assumed to be a Firedrake VertexOnlyMesh.

    Return finish time.
    """
    pmesh_updater = UpdateVertexOnlyMesh(pmesh, mesh)

    while t < T:
        coords = pmesh.coordinates 
        coords_io = pmesh.input_ordering.coordinates
        
        # v must live on the current VOM
        assert v.function_space().mesh() is pmesh

        # compute an adaptive time step so that particles remain in the domain
        # Here I put the dimensions of the unit square domain instead of the initial box where the particles started
        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
        dt_s = compute_safe_dt(coords.dat.data_ro, v.dat.data_ro, dt,
                               xmin, xmax, ymin, ymax)

        # stop if dt_s is zero (particle hitting boundary)
        if dt_s <= 1e-14:
            print(f"Stopping at t={t}: at least one particle reached the domain boundary.")
            break

        update_expr = coords + dt_s * v                 
        new_coords = assemble(interpolate(update_expr, coords.function_space()))
        new_coords_io = assemble(interpolate(update_expr, coords_io.function_space()))

        # make sure that interpolate is doing the update correctly
        # diff_step = new_coords.dat.data_ro - (coords.dat.data_ro + dt_s * v.dat.data_ro)
        # print("per-step max diff:", np.max(np.abs(diff_step)))

        # update the VOM
        pmesh_updater.update(new_coords_io.dat.data_ro)

        # redefine FS
        # V_new = VectorFunctionSpace(pmesh, "DG", 0, dim=2)
        # v_new = Function(V_new)
        # v_new.dat.data[:] = v.dat.data[:]
        # v = v_new

        # V_new = VectorFunctionSpace(pmesh, "DG", 0, dim=2)
        # v = Function(V_new).interpolate(v)

        # TODO: Permute the particle velocities after updating the VOM 
        # since updating the VOM changes the vertex numbering
        # new_coords[i] is the position of some new vertex j,
        # v[i] is still the velocity of the old vertex i (velocities are attached to wrong particles!)

        t += dt_s 

    return t

def compute_safe_dt(coords, velocities, dt_default, x_min, x_max, y_min, y_max):
    """
    Given the particles positions and velocities and the dimensions of the domain box, 
    compute dt <= dt_default such that no particle exits the domain.
    """

    safe_dt = dt_default
    for p, v in zip(coords, velocities):
        for coord, vel, low, high in [(p[0], v[0], x_min, x_max), (p[1], v[1], y_min, y_max)]:
            if vel > 0:
                dt_max = (high - coord) / vel
                safe_dt = min(safe_dt, dt_max)
            elif vel < 0:
                dt_max = (coord - low) / abs(vel)
                safe_dt = min(safe_dt, dt_max)

    return float(max(safe_dt, 0.0))


# Define the parent mesh
# increase the mesh resolution to reduce the tolerance of the error in domain approx.
mesh = UnitSquareMesh(10, 10)
x = SpatialCoordinate(mesh)

# Define the particles in a VOM
N = 10

# Define a box within the mesh to place the particles in initially
xmin, xmax = 0.2, 0.8
ymin, ymax = 0.2, 0.8

# particle_coords = np.random.rand(N, 2) 
particle_coords = np.zeros((N, 2))
particle_coords[:, 0] = xmin + (xmax - xmin) * np.random.rand(N)
particle_coords[:, 1] = ymin + (ymax - ymin) * np.random.rand(N)

particle_vom = VertexOnlyMesh(mesh, particle_coords)
print("Initial particle positions (in primary VOM ordering): ", particle_vom.coordinates.dat.data)

# Save initial positions in VOM ordering for exact solution
x0_vom = particle_vom.coordinates.dat.data_ro.copy()

# Assign per-particle velocities
P0DG = VectorFunctionSpace(particle_vom, "DG", 0, dim=2)
P0DG_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=2)

# Following the steps in PointEvaluator: assign velocities to the input ordering VOM then interpolate to primary VOM
input_velocities = np.random.normal(0.0, 0.5, size=(N,2))
v_io = Function(P0DG_io)
v_io.dat.data[:] = input_velocities
v_vom = Function(P0DG)
v_vom.interpolate(v_io)

# v_vom = Function(P0DG)
# velocities = np.random.normal(0.0, 0.5, size=(N,2))
# v_vom.dat.data[:] = velocities # NOTE: this is wrong since VOM's internal ordering does not match input ordering

# Save initial velocities in VOM ordering for exact solution
v0_vom = v_vom.dat.data_ro.copy()

# Wrap the particles VOM in a rebuilder class
# particle_vom_rebuilder = RebuildVertexOnlyMesh(particle_vom, mesh)

# NOTE: By interpolating the global coordinate field into the the particle mesh coordinate field 
# we notice a discrepancy in the positions of the particles.
# This is due to VOM being a static structure: updating its vertex coordinates does not trigger an update of its internal topological structure
# which means that the mapping vertex index -> parent cell num, ref coords, parallel ownership etc. are all invalid and by extension, 
# any function spaces defined on the VOM are invalid too (their DoF layout uses incorrect vertex indices)

# Move particles using the VOM rebuilder wrapper
# particles_updated, T_final = move_particles_and_rebuild_vom(particle_vom_rebuilder, v_vom, dt, T, t=0.0)
# print("Final particle positions (updated particle VOM): ", particles_updated.coordinates.dat.data_ro)

# NOTE: The *newly* created VOM sometimes returns an empty array of points or less particles than what we started with.
#  This occurs if the updated particle positions could not be located within the parent mesh cells which leads to their exclusion from the DMSwarm 
# (via the rule parent_cell_nums_local != -1). Importantly, this is not a hard rule in the sense that some particles with coords. clearly outside the domain may still appear.

# Move particles using the same VOM
# dt = T
T_final = move_particles_and_update_vom(particle_vom, mesh, v_vom, dt, T, t=0.0)
print("Final particle positions (in updated VOM ordering): ", particle_vom.coordinates.dat.data)

# Confirm that final coords match parent mesh interpolation
embedding_func = assemble(interpolate(x, particle_vom.coordinates.function_space()))
# print("Parent mesh embedding: ", embedding_func.dat.data)
print(embedding_func.dat.data - particle_vom.coordinates.dat.data)

# Check integration error
# With constant velocities, forward Euler must be exact
exact_final_coords = x0_vom + T_final * v0_vom
print("Exact final particle positions (in primary VOM ordering): ", exact_final_coords)
 
error = particle_vom.coordinates.dat.data - exact_final_coords
print("Integration error (L^2): ", np.linalg.norm(error))

# NOTE: The above error may be comparing particles in the wrong order!
# Inspect permutation issue
# final_coords_sorted_x = np.sort(particle_vom.coordinates.dat.data[:, 0])
# final_coords_exact_sorted_x = np.sort(exact_final_coords[:, 0])
# print(np.max(np.abs(final_coords_sorted_x - final_coords_exact_sorted_x)))
