from firedrake import *
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from particle_crossing_solver import BisectionSolver
from particle_time_stepper_new import ForwardEulerStepper
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

# ---- particle VTK writers ---
def write_particle_vtk(fname, vom, function, coordinates=None):
    if coordinates is None:
        coordinates = vom.coordinates.dat.data_ro
    coordinates = np.pad(coordinates, ((0, 0), (0, 1)))   # (N,2)->(N,3): add z=0 column only
    dat = np.pad(function.dat.data_ro, ((0, 0), (0, 1)))
    num_points = coordinates.shape[0]

    def descr(f, name, arr, offset):
        ncmp = {0: "", 1: "3"}[len(arr.shape[1:])]
        typ = {np.dtype("float64"): "Float64"}[arr.dtype]
        f.write(f'<DataArray Name="{name}" type="{typ}" NumberOfComponents="{ncmp}" format="appended" offset="{offset}" />'.encode())
        return 4 + arr.nbytes

    def warr(f, arr):
        np.uint32(arr.nbytes).tofile(f); arr.tofile(f)

    with open(fname, "wb") as f:
        off = 0
        f.write(f'''<?xml version="1.0" ?>
<VTKFile type="UnstructuredGrid" version="2.2" byte_order="LittleEndian" header_type="UInt32">
<UnstructuredGrid>
<Piece NumberOfPoints="{num_points}" NumberOfCells="0">
<Cells>
<DataArray type="Int32" Name="connectivity" format="ascii"></DataArray>
<DataArray type="Int32" Name="offsets" format="ascii"></DataArray>
<DataArray type="UInt8" Name="types" format="ascii"></DataArray>
</Cells>
<Points>
'''.encode())
        off += descr(f, "Coordinates", coordinates, off)
        f.write(b"</Points><PointData Vectors=\"Velocity\">")
        off += descr(f, "Velocity", dat, off)
        f.write(b'''</PointData>
</Piece>
</UnstructuredGrid>
<AppendedData encoding="raw">
_''')
        warr(f, coordinates); warr(f, dat)
        f.write(b"</AppendedData>\n</VTKFile>")


def write_pvd(fname, tmpl, count):
    with open(fname, "w") as f:
        f.write('<?xml version="1.0" ?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for i in range(count):
            f.write(f'<DataSet timestep="{i}" file="{tmpl.format(i)}" />\n')
        f.write("</Collection></VTKFile>")


mesh = UnitSquareMesh(40, 40, quadrilateral=False)
V = VectorFunctionSpace(mesh, "CG", 2) # velocity FS

num_particles = 10
r = 0.25
c = np.array([0.5, 0.5])
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)
q0 = np.column_stack([c[0] + r*np.cos(theta), c[1] + r*np.sin(theta)])

particle_vom = VertexOnlyMesh(mesh, q0)
P0DG = VectorFunctionSpace(particle_vom, "DG", 0)

# Define the Taylor-Green vortex (steady)
U0 = 40.0 # fluid speed -> determines how many cell crossings occur per time step
x, y= SpatialCoordinate(mesh)
u_expr = U0 * as_vector([sin(pi*x)*cos(pi*y),
                         -cos(pi*x)*sin(pi*y)])

# Store the velocity field in a Function to plot streamlines in Paraview
# (curves tangent to the velocity field everywhere)
V = VectorFunctionSpace(mesh, "CG", 2)
u_fn = Function(V, name="Velocity")
u_fn.interpolate(u_expr)

# --- Solvers ---
dt, timesteps = 1e-3, 200
# stepper = ForwardEulerStepper(particle_vom, dt, u_fn)
stepper = ForwardEulerStepper(particle_vom, dt, u_expr)
params = ParticleTrajectorySolverParams(bary_tol=1e-9, abs_time_tol=1e-9,
                                        rel_time_tol=0, max_iters=50, plot=False)
solver = ParticleTrajectorySolver(stepper, BisectionSolver(), params)

# ---- output ----
VTKFile("output.pvd").write(u_fn)
Path("particles").mkdir(exist_ok=True)
base = "particles/particles_{}.vtu"
count = 0

# Evaluate the velocity field at the particles
u_at_points = Function(P0DG)
u_at_points.interpolate(u_fn)

time = 0.0
for timestep in range(timesteps):
    write_particle_vtk(base.format(count), particle_vom, u_at_points)
    count += 1

    time_start = time
    stepper._dt = dt
    time += dt
    # u_fn.interpolate(u_expr)
    solver.solve(time_start, time)
    u_at_points.interpolate(u_fn)

write_pvd("particles.pvd", base, count)