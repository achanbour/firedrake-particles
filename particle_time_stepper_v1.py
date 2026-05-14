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
from ufl.domain import extract_unique_domain

class ParticleTimeStepper(ABC):
    """
    Abstract base class for time steppers that numerically integrate an ODE
    describing a particle trajectory dx/dt = v.
    """
    def __init__(self, particle_vom, dt, v, **kwargs):
        # NOTE: kwargs contain stepper parameter e.g., tolerances for error refinement in adapative time stepping
        # or Butcher tableau for describing the integrator's parameters.

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
        self._fields = [self._dt_fn] # register fields owned by the stepper

        
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
    
    # NOTE: Using private attributes + property methods enforce a read-only contract on fields owned by the stepper.
    # Re-assigning the fields with a new Function (e.g., stepper.X = ...) would silently invalidate the update expr and associated cached parloops.
    # If we want to allow users to swap out the fields then this should be done via an explicit setter method:

    # @dt.setter
    # def dt(self, val):
    #     self._dt = val
    #     self.update_expr = self._build_update_expr()
    #     self.invalidate() # call invalidate to force parloops to be rebuilt


    def _build_step_callable(self):
        """Build and cache the step interpolation callables."""
        interpolation_expr = interpolate(self._update_expr, self._X.function_space()) # symbolic interpolation expr
        interpolator = get_interpolator(interpolation_expr) # numerical interpolator
        self.step_callable = interpolator._get_callable() # callable that executes the parloops
        self._step_callable_is_current = True
    

    def _check_step_callable_is_current(self):
        """Trigger a rebuild of the step interpolation callables."""
        if not self._step_callable_is_current:
            self._build_step_callable()


    def _rebuild_fields(self):
        """Rebuild all fields eagerly after a change to the VOM's topology"""
        for f in self._fields:
            if isinstance(f, Function):
                f._match_mesh_topology_version() # rebuilds the FS
            else:
                raise TypeError("Encountered a field in the stepper that is not a Function.")
    

    def _rebuild_exprs(self):
        """Rebuild all expressions that form the update expression"""
        pass

    def _reevaluate_fields(self):
        """Re-evaluate all fields that form the update expression"""
        pass


    # NOTE: To be removed once rebuild_vom and rebuild_function are fixed.
    def invalidate(self):
        """Mark all callables as stale."""
        self._rebuild_fields() # migrates the Functions' data and swaps their FS
        self._rebuild_exprs() # reconstruct the interpolation expression to reference the new FS
        self._build_update_expr() # new interpolate node implies the UFL expression needs to be reconstructed
        self._step_callable_is_current = False


    def step(self):
        # self._reevaluate_fields
        self._check_step_callable_is_current()
        result = self.step_callable() # execute the cached parloops
        return result


class ForwardEulerStepper(ParticleTimeStepper):
    """
    Advance particles by Forward Euler:

    X(t + dt) = X(t) + J^-1 * v * dt

    where J is the Jacobian of the geometric map from reference space to physical space F: X -> x
    used to pullback the velocity field to reference space.
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

        # Option 1: Define and store the symbolic expression for v_ref and use that in the update expression
        if extract_unique_domain(self._v) is self.particle_vom:
            TFS = TensorFunctionSpace(self.particle_vom, "DG", 0)

            # NOTE: TFS becomes stale when particle_vom is rebuilt!
            # Hence we register the fields so their FS get rebuilt.
            self.invJ_fn = Function(TFS)
            self._fields.append(self.invJ_fn)
            # self._v_ref = interpolate(self._invJ_expr, TFS) * self._v
            self._v_ref = interpolate(self._invJ_expr, self.invJ_fn.function_space()) * self._v
        else:
            VFS = VectorFunctionSpace(self.particle_vom, "DG", 0)
            self._v_ref_fn = Function(VFS)
            self._fields.append(self._v_ref_fn)
            # self._v_ref = interpolate(self._invJ_expr, VFS) * self._v
            self._v_ref = interpolate(self._invJ_expr * self._v, self._v_ref_fn.function_space())

        """
        # Option 2: Define separate expressions for invJ and v and use their assembled output (Function) in the update expression
        TFS = TensorFunctionSpace(self.particle_vom, "DG", 0)
        self._invJ_fn = Function(TFS)
        self._fields.append(self._invJ_fn)

        VFS = VectorFunctionSpace(self.particle_vom, "DG", 0)
        self._v_vom = Function(VFS)
        self._fields.append(self._v_vom)

        self._v_vom.interpolate(self._v)

        # Both functions are on the VOM so the product is defined on a single domain.
        self._v_ref = self._invJ_fn * self._v_vom
        """
        
        self._build_update_expr()
        self._build_step_callable()


    def _build_update_expr(self):
        self._update_expr = self._X + self._v_ref * self._dt_fn


    def _reevaluate_fields(self):
        self._invJ_fn.interpolate(self._invJ_expr)
        self._v_vom.interpolate(self._v)

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
        # return assemble(interpolate(self._v_ref, self._v_vom.function_space()))
        return assemble(interpolate(self._v_ref, self.particle_vom.coordinates.function_space()))


"""
evaluating the update expr amounts to doing assemble(interpolate(update_expr, VOM)) where update expr is: 
X + dt * v_ref where v_ref = invJ * v

- invJ = inv(Reference(Grad(x))) lives on parent mesh
- v may live on VOM or on parent mesh
- so v_ref = interpolate(invJ * v, VOM) or interpolate(invJ, VOM) * v

instead of assembling v_ref into a Function (which involves assembles two interpolations), keep v_ref symbolic in the update expr.

currently fails in get_interpolator as extract_unique_domain raises an error

interpolate(interpolate(invJ, VOM) * v, VFS) assembles into a Coefficient on the VOM

Try splitting the operand using CoefficientSplitter(interpolate(invJ, VOM) * v) as is done in
MixedInterpolator on forms using arguments.
"""