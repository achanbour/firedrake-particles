from abc import ABC, abstractmethod

from firedrake.interpolation import interpolate, get_interpolator
from firedrake.assemble import assemble
from firedrake.mesh import VertexOnlyMeshTopology
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace

from ufl.differentiation import ReferenceGrad
from ufl.geometry import SpatialCoordinate
from ufl.operators import inv
from ufl.core.expr import Expr

class ParticleTimeStepper(ABC):
    """
    Abstract base class for time steppers that numerically integrate an ODE
    describing a particle trajectory.
    """
    def __init__(self, particle_vom, dt, *args):
        if not isinstance(particle_vom.topology, VertexOnlyMeshTopology):
            raise TypeError("Expected particles to be represented as a VertexOnlyMesh.")

        self.particle_vom = particle_vom

        self._X = self.particle_vom.reference_coordinates

        if not isinstance(dt, float):
            raise TypeError("Expected the time step parameter dt to be a float.")

        self._dt = dt
        FS = FunctionSpace(particle_vom, "DG", 0)
        self._dt_fn = Function(FS, name="per-particle-time-step-function")

        self._fields = [self._dt_fn]
        self._setup_fields(*args)
        self._update_expr = self._build_update_expr()
        self._build_step_callable()

    @abstractmethod
    def _setup_fields(self, *args):
        """Store stepper-specific fields."""
        for val in args:
            if not isinstance(val, (Function, Expr)):
                raise TypeError(f"Expected each field to be Firedrake Function or a UFL expression.")

    @abstractmethod
    def _build_update_expr(self):
        """Return the UFL update expression."""
        pass

    @property
    def X(self):
        return self._X

    @property
    def dt_fn(self):
        return self._dt_fn

    @property
    def dt(self):
        return self._dt

    # NOTE: private attributes + property methods enforce a read-only contract on fields.
    # Re-assigning the fields with a new Function (e.g., stepper.X = ...) would silently invalidate the update expr and associated cached parloops.
    # If we want to allow users to swap out the fields then this should be done via an explicit setter method:

    # @dt.setter
    # def dt(self, val):
    #     self._dt = val
    #     self.update_expr = self._build_update_expr()
    #     self.invalidate() # call invalidate to force parloops to be rebuilt

    @property
    @abstractmethod
    def v_ref(self):
        """Return the pullback of the velocity field to reference space"""
        pass

    def _build_step_callable(self):
        """Build and cache the interpolation callables."""
        interpolation_expr = interpolate(self._update_expr, self._X.function_space()) # symbolic interpolation expr
        interpolator = get_interpolator(interpolation_expr) # numerical interpolator
        self.step_callable = interpolator._get_callable() # parloops
        self._step_callable_is_current = True

    def _check_step_callable_is_current(self):
        """Trigger a rebuild of the interpolation callables."""
        if not self._step_callable_is_current:
            self._build_step_callable()

    def _reevaluate_fields(self):
        """Re-evaluate all fields needed by step."""
        pass

    def _rebuild_fields(self):
        """Rebuild all fields eagerly after a change to the VOM's topology"""
        for f in self._fields:
            if isinstance(f, Function):
                f._match_mesh_topology_version()

    # NOTE: To be removed once rebuild_vom is fixed.
    def invalidate(self):
        """Mark all callables as stale."""
        self._rebuild_fields() # either called externally or as part of invalidate
        self._step_callable_is_current = False

    def step(self):
        self._reevaluate_fields()
        self._check_step_callable_is_current()
        result = self.step_callable() # execute cached parloops
        return result


class ForwardEulerStepper(ParticleTimeStepper):
    """
    Advance particles by Forward Euler:

    X(t + dt) = X(t) + J^-1 * v * dt

    where J is the Jacobian of the geometric map from reference space to physical space F: X -> x
    used to pullback the velocity field to reference space.
    """
    def _setup_fields(self, *args):
        super()._setup_fields(*args)
        v = args[0] # expect only velocity to be passed

        VFS = VectorFunctionSpace(self.particle_vom, "DG", 0)
        self._v_vom = Function(VFS)

        if isinstance(v, Function) and v.function_space().mesh() is self.particle_vom:
            # velocity already lives on the VOM - copy once, no re-evaluation needed
            self._v = v
            self._v_vom = v
        else:
            # velocity is a UFL expression or a Function on an external mesh — re-interpolate each step
            self._v = v
            self._v_vom.interpolate(v)

        self._fields.append(self._v_vom)

        # Additional fields owned by the stepper
        x = SpatialCoordinate(self.particle_vom._parent_mesh)
        self._invJ_expr = inv(ReferenceGrad(x))

        TFS = TensorFunctionSpace(self.particle_vom, "DG", 0)
        self._invJ_fn = Function(TFS)
        self._fields.append(self._invJ_fn)

        self._v_ref = Function(VFS)
        self._fields.append(self._v_ref)
        self._v_ref_callable_current = False

    def invalidate(self):
        super().invalidate()
        # Option 1: mark callable as stable, gets rebuilt lazily on next access
        self._v_ref_callable_current = False

        # Option 2: force the callable to be rebuilt now
        # self._build_v_ref_callable()

        # NOTE: whether we rebuild v_ref's callable before or after _invJ_fn is re-interpolated 
        # doesn't matter in practice, because the parloop for v_ref is only executed when v_ref is accessed, 
        # at which point _invJ_fn has already been re-evaluated.

    def _build_v_ref_callable(self):
        _v_ref_interpolator = get_interpolator(
            interpolate(self._invJ_fn * self._v_vom, self._v_ref.function_space())
        )
        self._v_ref_callable = _v_ref_interpolator._get_callable()
        self._v_ref_callable_current = True

    def _build_update_expr(self):
        return self._X + self._invJ_fn * self._v_vom * self._dt_fn

    def _reevaluate_fields(self):
        # NOTE: do we need to cache the interpolation callables too?
        # Likely yes as this is called every time stepper.step() including the calls made in bisection
        # which would repeatedly re-evaluate the fields!
        self._invJ_fn.interpolate(self._invJ_expr)

        # If v is an external field then we need to re-evaluate at the new particle locations
        if self._v is not self._v_vom:
            self._v_vom.interpolate(self._v)

    @property
    def v(self):
        return self._v

    @property
    def v_ref(self):
        if self._v_ref_callable_current is False:
            self._build_v_ref_callable()
        result =  self._v_ref_callable() # execute cached parloops
        return result