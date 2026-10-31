from firedrake import *
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

from particle_time_stepper import StormerVerletStepper
from particle_crossing_solver import BisectionSolver, BisectionSolverParams
from particle_traj_solver import ParticleTrajectorySolver, ParticleTrajectorySolverParams

from output_paths import experiment_results_dir
import matplotlib.pyplot as plt

c, r, omega, num_particles = np.array([0.5, 0.5]), 0.25, 10*np.pi, 10
theta = np.linspace(0, 2*np.pi, num_particles, endpoint=False)
t_end = 1.0
n_steps = [100, 200, 400, 800, 1600, 2400] # dt = t_end/n_steps
solver_params = particle_traj_solver_params = ParticleTrajectorySolverParams(
    bary_tol=1e-9,
    abs_time_tol=1e-9,
    rel_time_tol=0,
    max_iters=50,
    plot=False
)

# initial positions starting on a circle centered at c with radius r
x0 = c[0] + r*np.cos(theta)
y0 = c[1] +r*np.sin(theta)
q0 = np.column_stack([x0, y0])

def make_parent_mesh(is_curved=False):
    base = UnitSquareMesh(10, 10, quadrilateral=False)
    if is_curved == False:
        return base
    else:
        V = VectorFunctionSpace(base, "CG", 3)
        v = Function(V)
        x, y = SpatialCoordinate(base)
        x_new = x + 0.5*x**2*(1-x)
        y_new = y + 0.5*y**2*(1-y)
        v.interpolate(as_vector([x_new, y_new]))
        curved = Mesh(v)
        return curved

def run(is_mesh_curved, is_stepper_corrected, dt):
    mesh = make_parent_mesh(is_mesh_curved)

    # Define parent mesh fields
    x = SpatialCoordinate(mesh)
    displacement = as_vector([x[0] - c[0], x[1] - c[1]])
    potential_expr = 0.5*omega**2*inner(displacement, displacement)
    accel_expr = -grad(potential_expr)

    vom = VertexOnlyMesh(mesh, q0)
    q0_vom = vom.coordinates.dat.data_ro.copy()
    v0_vom = 0.5 * omega * np.column_stack([-(q0_vom[:, 1] - c[1]), q0_vom[:, 0] - c[0]])

    stepper = StormerVerletStepper(vom, dt, accel_expr, geometric_correction=is_stepper_corrected)
    stepper.v.dat.data_wo[:] = v0_vom

    solver = ParticleTrajectorySolver(stepper, BisectionSolver(), solver_params)
    solver.solve(0.0, t_end)

    surviving_particle_ids = vom._particle_ids.dat.data_ro

    # assert that no particle is lost at the end of the simulation
    assert len(surviving_particle_ids) == num_particles, "particles were lost at the boundary"

    # check discrepancy against analytical solution
    exact  = c + np.cos(omega*t_end) * (q0_vom - c) + np.sin(omega*t_end) / omega * v0_vom
    return np.max(np.linalg.norm(vom.coordinates.dat.data_ro - exact, axis=1))

errors = {} # key: (is_mesh_curved, is_stepper_corrected), values: list of errors, one per dt
for is_mesh_curved in [False, True]:
    for is_stepper_corrected in [False, True]:
        errors[is_mesh_curved, is_stepper_corrected] = [run(is_mesh_curved, is_stepper_corrected, t_end/n) for n in n_steps]

# Diagnostics
# print: dt, error, observed order log2(e[i]/e[i+1])
dts =np.array([t_end/n for n in n_steps])
def case_label(is_mesh_curved, is_stepper_corrected):
    return f"curved: {is_mesh_curved}, sv corrected: {is_stepper_corrected}"

for (is_mesh_curved, is_stepper_corrected), errs in errors.items():
    errs = np.asarray(errs)
    orders = np.log(errs[:-1]/errs[1:]) / np.log(dts[:-1] / dts[1:])

    print(f"\n{case_label(is_mesh_curved, is_stepper_corrected)}")
    print(f"{'dt':>10} {'error':>12} {'order':>7}")
    for i in range(len(dts)):
        order = f"{orders[i-1]:7.2f}" if i > 0 else f"{'-':>7}"
        print(f"{dts[i]:10.2e} {errs[i]:12.4e} {order}")

# log-log plot error vs dt: err ~= C x dt^p -> log(err) = log(C) + p*log(dt)
styles = {
    (False, False): dict(marker="o", linestyle="-"),
    (False, True):  dict(marker="x", linestyle="--"),
    (True, False):   dict(marker="s", linestyle="-"),
    (True, True):    dict(marker="^", linestyle="--"),
}

fig, ax = plt.subplots()
for key, errs in errors.items():
    ax.loglog(dts, errs, label=case_label(*key), **styles[key])

ax.set_xlabel("dt")
ax.set_ylabel("max position error at t_end")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
output_dir = experiment_results_dir(__file__)
fig.savefig(output_dir / "sv_convergence_plot.png", dpi=3000)

# log-log plot shows that the corrected SV stepper is second-order on the curved mesh and the errors match the straight mesh case for all dts tested.
# Without the correction, SV on the curved mesh converges at first order.
# The order can be inferred from log(error ratio)/log(dt ratio) for nearby points on the plot (under the assumption that the error behaves like a power of dt).