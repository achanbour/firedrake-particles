from firedrake import *
import time
import numpy as np

mesh = UnitSquareMesh(500, 500)
x, y = SpatialCoordinate(mesh)
expr = sin(pi*x)*sin(pi*y)

V = FunctionSpace(mesh, "CG", 3)
# Run once to cache compiled kernel, generated code from parloops, etc.
res1 = assemble(interpolate(expr, V))

t0 = time.perf_counter_ns()
res1 = assemble(interpolate(expr, V))
t1 = time.perf_counter_ns()
print(f"Assembly took {(t1-t0)*1e-9:.6f} seconds")

I = get_interpolator(interpolate(expr, V))
f = I._get_callable()

t0 = time.perf_counter_ns()
res2 = f()
t1 = time.perf_counter_ns()
print(f"Callable took {(t1-t0)*1e-9:.6f} seconds")

assert np.allclose(res1.dat.data, res2.dat.data)