from abc import ABC, abstractmethod
from functools import partial

from firedrake.interpolation import interpolate, get_interpolator
from firedrake.assemble import assemble
from firedrake.mesh import VertexOnlyMeshTopology
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace

from ufl.differentiation import ReferenceGrad
from ufl.geometry import SpatialCoordinate
from ufl.operators import inv
from ufl.core.expr import Expr
from ufl.domain import extract_unique_domain


class ParticleTimeStepper(ABC):
    """
    Abstract base class for time steppers that numerically integrate an ODE
    describing a particle trajectory e.g., dx/dt = v.
    """
    def __init__(self, particle_vom, dt, v, **kwargs):
        # NOTE: kwargs contain stepper parameter e.g., tolerances for error refinement in adapative time stepping
        # or Butcher tableau describing the integrator's parameters.

        # Validate the stepper's arguments
        if not isinstance(particle_vom.topology, VertexOnlyMeshTopology):
            raise TypeError("Expected particles to be given as a VertexOnlyMesh.")
        
        if not isinstance(v, Expr):
            raise TypeError("Expected velocity to be given as a UFL expression.")

        self.particle_vom = particle_vom
        self._X = self.particle_vom.reference_coordinates

        if not isinstance(dt, float):
            raise TypeError("Expected the time step parameter to be given as a float.")
        
        # Store the time step as a particle field
        self._dt = dt
        FS = FunctionSpace(particle_vom, "DG", 0)
        self._dt_fn = Function(FS, name="per-particle-time-step-function")
        
        # Register fields owned by stepper
        self._fields = [self._dt_fn] 

        
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
    
    # NOTE: Using private attributes + property methods enforce a read-only contract on fields that form the stepper's update expression.
    # Re-assigning the fields with a new Function (e.g., stepper.X = ...) would silently invalidate the update expr and associated cached parloops.
    # If we want to allow users to swap out the fields then this should be done via an explicit setter method:

    # @dt.setter
    # def dt(self, val):
    #     self._dt = val
    #     self.update_expr = self._build_update_expr()
    #     self.invalidate() # call invalidate to force parloops to be rebuilt


    def _build_step_callable(self):
        """Build and cache the step interpolation callable."""
        interpolation_expr = interpolate(self._update_expr, self._X.function_space()) # symbolic interpolation expr
        self.step_callable = partial(assemble, interpolation_expr)
        self._step_callable_is_current = True


    def _check_step_callable_is_current(self):
        """Rebuild the step interpolation callable."""
        if not self._step_callable_is_current:
            self._build_step_callable()


    def _rebuild_fields(self):
        """Rebuild all fields after a VOM state change"""
        for f in self._fields:
            if isinstance(f, Function):
                f._match_mesh_topology_version()
            else:
                raise TypeError("Cannot rebuild a field that is not a Function.")
    
    def _rebuild_exprs(self):
        """Rebuild all expressions"""
        pass


    # NOTE: To be removed once rebuild_vom and rebuild_function are fixed.
    def invalidate(self):
        """Mark all callables as stale."""
        self._rebuild_fields() # migrates the Functions' data and swaps their FS
        self._rebuild_exprs() # reconstruct the interpolation expression to reference the new FS
        self._build_update_expr() # new interpolate node implies the UFL expression needs to be reconstructed
        self._step_callable_is_current = False


    def step(self):
        self._check_step_callable_is_current()
        result = self.step_callable() # assemble the cached interpolation
        return result


class ForwardEulerStepper(ParticleTimeStepper):
    """
    Advance particles by Forward Euler:

    X(t + dt) = X(t) + J^-1 * v * dt

    where J is the Jacobian of the geometric map from reference space to physical space F: X -> x
    used to pull the velocity field back to reference space.
    """

    def __init__(self, particle_vom, dt, v):
        super().__init__(particle_vom, dt, v)

        # Define any additional fields owned by the stepper
        x = SpatialCoordinate(self.particle_vom._parent_mesh)
        self._invJ_expr = inv(ReferenceGrad(x))
        
        self._v = v

        if extract_unique_domain(v) is self.particle_vom:
            # Appending self._v to self._fields ensures that it gets resized appropriately to match the new topology
            # so that the interpolation to self._v is correct
            self._fields.append(self._v)

        # Define and store the symbolic expression for v_ref to be used in the update expression
        # NOTE: The FS defined on the particle VOM may become stale when the particle VOM gets rebuilt!
        # Registering the fields ensures their underlying FS get rebuilt, and therefore that the target space in the interpolation
        # is always defined on the latest VOM.
        if extract_unique_domain(self._v) is self.particle_vom:
            TFS = TensorFunctionSpace(self.particle_vom, "DG", 0)
            # self._v_ref = interpolate(self._invJ_expr, TFS) * self._v
            self.invJ_fn = Function(TFS)
            self._fields.append(self.invJ_fn)
            self._v_ref = interpolate(self._invJ_expr, self.invJ_fn.function_space()) * self._v
        else:
            VFS = VectorFunctionSpace(self.particle_vom, "DG", 0)
            # self._v_ref = interpolate(self._invJ_expr * self._v, VFS)
            self._v_ref_fn = Function(VFS)
            self._fields.append(self._v_ref_fn)
            self._v_ref = interpolate(self._invJ_expr * self._v, self._v_ref_fn.function_space())
        
        self._build_update_expr()
        self._build_step_callable()


    def _build_update_expr(self):
        self._update_expr = self._X + self._v_ref * self._dt_fn

    def _rebuild_exprs(self):
        if extract_unique_domain(self._v) is self.particle_vom:
            self._v_ref = interpolate(self._invJ_expr, self.invJ_fn.function_space()) * self._v
        else:
            self._v_ref = interpolate(self._invJ_expr * self._v, self._v_ref_fn.function_space())

    @property
    def v(self):
        return self._v
    
    @property
    def v_ref(self):
        # TODO: Cache the result from step and reuse
        return assemble(interpolate(self._v_ref, self.particle_vom.coordinates.function_space()))