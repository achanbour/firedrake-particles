from firedrake.interpolation import interpolate, get_interpolator
from firedrake import Function

STEP_COUNT = 0

class ForwardEulerTimeStepper:
    def __init__(self, X, invJ, v, dt):
        self._X = X
        self._invJ = invJ
        self._v = v
        self._dt = dt

        # All terms are assumed to be Functions
        # Check they're all defined on the same mesh (particle VOM)
        m = X.function_space().mesh()
        assert invJ.function_space().mesh() == m
        assert dt.function_space().mesh() == m

        # NOTE: in general, v could be an expression on the VOM OR parent mesh
        # For now, we assume it's a function of the VOM
        assert v.function_space().mesh() == m
        
        # Forward Euler update expression (in ref. space)
        self.update_expr = X + invJ * v * dt

        self.interp_expr = None
        self.interpolator = None
        self.callable = None

        self._build_callable()

    @property
    def X(self):
        return self._X
    
    @property
    def invJ(self):
        return self._invJ
    
    @property
    def v(self):
        return self._v
    
    @property
    def dt(self):
        return self._dt

    @property
    def update_expr(self):
        return self._update_expr

    @X.setter
    def X(self, value):
        self._X = value
        self._rebuild_update_expr()

    @invJ.setter
    def invJ(self, value):
        self._invJ = value
        self._rebuild_update_expr()

    @v.setter
    def v(self, value):
        self._v = value
        self._rebuild_update_expr()

    @dt.setter
    def dt(self, value):
        self._dt = value
        self._rebuild_update_expr()

    def _rebuild_update_expr(self):
        self.update_expr = self._X + self._invJ * self._v * self._dt
        
    @update_expr.setter
    def update_expr(self, value):
        self._update_expr = value
        self.invalidate()
    
    def invalidate(self):
        self._callable_is_current = False

    def _build_callable(self):
        self.interp_expr = interpolate(self.update_expr, self.X.function_space()) # symbolic interpolation expr
        self.interpolator = get_interpolator(self.interp_expr) # numerical interpolator
        self.callable = self.interpolator._get_callable() # parloops
        self._callable_is_current = True

    def _check_callable_is_current(self):
        if not self._callable_is_current:
            self._build_callable()

    def step(self):
        # global STEP_COUNT
        # STEP_COUNT += 1
        
        self._check_callable_is_current()

        # Execute cached ParLoops
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
