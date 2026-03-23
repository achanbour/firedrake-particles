from firedrake.interpolation import interpolate, get_interpolator
from firedrake import Function

STEP_COUNT = 0

class ForwardEulerTimeStepper:
    def __init__(self, X, invJ, v, dt):
        self.X = X
        self.invJ = invJ
        self.v = v
        self.dt = dt

        V = X.function_space()
        self.V = V

        # All terms are assumed to be Functions
        # So we check they're all defined on the same mesh (particle VOM)
        m = V.mesh()
        assert invJ.function_space().mesh() == m
        assert dt.function_space().mesh() == m

        # v could be an expression on the VOM OR parent mesh
        assert v.function_space().mesh() == m
        
        # Forward Euler update expression (in ref. space)
        self.update_expr = X + invJ * v * dt

        self.output = Function(V)

        self.interp_expr = None
        self.interpolator = None
        self.callable = None
        self._build_callable()

    @property
    def update_expr(self):
        return self._update_expr

    @update_expr.setter
    def update_expr(self, expr):
        self._update_expr = expr
        self._callable_is_current = False

    def _build_callable(self):
        self.interp_expr = interpolate(self.update_expr, self.V) # symbolic interpolation expr
        self.interpolator = get_interpolator(self.interp_expr) # numerical interpolator
        self.callable = self.interpolator._get_callable(tensor=self.output)
        self._callable_is_current = True

    def _check_callable_is_current(self):
        if not self._callable_is_current:
            self._build_callable()

    def step(self):
        global STEP_COUNT
        STEP_COUNT += 1
        
        self._check_callable_is_current()

        # Execute existing ParLoops
        result = self.callable()
        return result

# NOTE 1:
# With the above persistent time stepper, we eliminate the overhead from symbolically reconstructing the interpolation expression
# and associated expensive TSFC re-compilation and ParLoop re-construction.
# This however does not reduce the number of runtime cache lookups that PyOP2 performs when exeucting the ParLoops.

# Updating the particles positions amounts to:
# 1. Defining the time stepper once
#   Per outer time loop or once per integration?
#   The VOM gets resized between successive time steps which causes the Function Spaces and Functions
#   to get automatically rebuilt. Since the underlying objects remain the same, this amounts to merely resizing the Dats
#   so we can most likely define the stepper outside the time loop.
# 2. Mutating the Dats of the Functions forming the update expression
# 3. Calling stepper.step()
    

# NOTE 2:
# In `_build_interpolation_callables()` called by `Interpolator._get_callable()`
# the line `parloop = op2.ParLoop(*parloop_args)` creates a ParLoop object without binding the backend kernel yet;
# this happens later when the ParLoop is executed.
# Executing a ParLoop (when calling the callable) triggers:
# ParLoop.__call__() -> ParLoop._compute() -> GlobalKernel.compile_global_kernel() at which point the kernel is looked up.
# PyOP2 essentially defers kernel binding until execution because that depends on the iteration partition (core/owned/halo), communicator state etc.
