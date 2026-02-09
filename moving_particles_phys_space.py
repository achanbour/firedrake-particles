from firedrake import *
import numpy as np
from update_vom import VertexOnlyMeshUpdater

np.random.seed(42)

# Define a time-integration scheme to move particles
t = 0.0
dt = 0.1
T = 1

def move_particles_in_phys_space(pmesh, mesh, v, dt, T, t=0.0):
    """
    Update particle positions in physical space using Forward Euler:
    
    x(t+dt) = x(t) + dt*v

    Args:
        pmesh: A VertexOnlyMesh containing the particles.
        mesh: The parent mesh.
        v: A DG(0) vector field on `pmesh` storing particle velocities.
        dt: Time step size.
        T: Final time up to which particles are integrated.
        t: Initial time.

    Returns:
        float: The final time reached when the time-stepping loop terminates.
    """
    pmesh_updater = VertexOnlyMeshUpdater(pmesh, mesh)

    while t < T:
        coords = pmesh.coordinates        
        
        # Ensure that the velocity field remains tied to the same particle VOM
        assert v.function_space().mesh() is pmesh
        
        # If not doing a single time step
        if dt != T:
            # Compute an adaptive time step so that particles remain in the domain
            # Below the dimensions of the unit square domain are used instead of the box where
            # the particles were initially placed
            xmin, xmax = 0, 1
            ymin, ymax = 0, 1
            dt = compute_safe_dt(coords.dat.data_ro, v.dat.data_ro, dt,
                                xmin, xmax, ymin, ymax)

            # stop if dt_s is zero (particle hitting boundary)
            if dt <= 1e-14:
                print(f"Stopping at t={t}: at least one particle reached the domain boundary.")
                break

        update_expr = coords + dt * v                 
        new_coords_fn = assemble(interpolate(update_expr, coords.function_space()))

        pmesh_updater.update(new_coords_fn)

        t += dt 

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

if __name__=='__main__':
    # Define the parent mesh
    # increase the mesh resolution to reduce the tolerance of the error in the domain approx.
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
    print("Initial particle positions (in input order): ", particle_coords)

    particle_vom = VertexOnlyMesh(mesh, particle_coords)
    print("Initial particle positions (in primary VOM order): ", particle_vom.coordinates.dat.data)

    # Save initial positions in VOM order for exact solution
    x0_vom = particle_vom.coordinates.dat.data_ro.copy()

    # Assign per-particle velocities
    P0DG = VectorFunctionSpace(particle_vom, "DG", 0, dim=2)
    P0DG_io = VectorFunctionSpace(particle_vom.input_ordering, "DG", 0, dim=2)

    # Following the steps in PointEvaluator:
    # assign velocities to the input ordering VOM then interpolate to primary VOM
    input_velocities = np.random.normal(0.0, 0.5, size=(N,2))
    v_io = Function(P0DG_io)
    v_io.dat.data[:] = input_velocities
    v_vom = Function(P0DG)
    v_vom.interpolate(v_io)

    # Save initial velocities in VOM order for exact solution
    v0_vom = v_vom.dat.data_ro.copy()

    # Set the parameters to do a single integration step
    T = 0.3
    dt = T 

    T_final = move_particles_in_phys_space(particle_vom, mesh, v_vom, dt, T, t=0.0)
    print("Final particle positions (in updated VOM order): ", particle_vom.coordinates.dat.data)

    # --- Check correctness of results ---
    # 1. Confirm that the parent mesh embedding has been updated.
    # i.e., that the coords obtained by interpolating the parent mesh into the updated VOM match the coords of the updated VOM
    embedding_func = assemble(interpolate(x, particle_vom.coordinates.function_space()))
    # print("Parent mesh embedding: ", embedding_func.dat.data)
    print("Parent mesh embedding diff: ", embedding_func.dat.data - particle_vom.coordinates.dat.data)

    # 2. Check the integration error
    # With constant velocities, forward Euler is exact (up to machine precision)
    exact_final_coords = x0_vom + T_final * v0_vom
    print("Exact final particle positions (in primary VOM order): ", exact_final_coords)

    error = particle_vom.coordinates.dat.data - exact_final_coords
    print("Integration error (L^2): ", np.linalg.norm(error))

# At the end of the integration, return two particle sets:
# 1. Particles with no remaining time -> Done
# 2. Particles with remaining time -> these must be moved

# Return tuple (new particle pos, remaining time, local ID of the cell face that the particle hit)
# use local ID to lookup the next cell in the topology information

# 2.a Particles that hit the rank partition boundary -> move to other process
# 2.b Particles that hit the domain boundary -> exclude from set
# 2.c Particles that hit internal boundary -> move to the adjacent cell


