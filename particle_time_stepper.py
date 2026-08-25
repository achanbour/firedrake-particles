from abc import ABC, abstractmethod
from functools import partial
from collections import namedtuple

from firedrake.interpolation import interpolate, Interpolate, get_interpolator
from firedrake.assemble import assemble
from firedrake.mesh import VertexOnlyMeshTopology
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace

from ufl.differentiation import ReferenceGrad
from ufl.geometry import SpatialCoordinate
from ufl.operators import inv
from ufl.core.expr import Expr
from ufl.domain import extract_unique_domain
from ufl.algorithms.analysis import extract_type


class ParticleTimeStepper(ABC):
    """
    Abstract base class for time steppers that numerically integrate a system of ODEs describing
    a particle trajectory.
    """
    def __init__(self, particle_vom, dt, **kwargs):
        # NOTE: kwargs is meant to contain additional stepper parameters 
        # e.g., tolerances for error refinement in adapative time stepping
        # or Butcher tableau describing the integrator's parameters.

        # Validate the stepper's arguments
        if not isinstance(particle_vom.topology, VertexOnlyMeshTopology):
            raise TypeError("Expected particles to be supplied as a VertexOnlyMesh.")
        
        if not isinstance(dt, float):
            raise TypeError("Expected the time step parameter to be given as a float.")

        self._dt = dt # user-assigned step size (float)
        self._dt_step = dt # size of the outer step currently in progress

        self.particle_vom = particle_vom
        self._X = self.particle_vom.reference_coordinates

        x = SpatialCoordinate(self.particle_vom._parent_mesh)
        self._invJ_expr = inv(ReferenceGrad(x)) # used for every position update

        # Define a per-particle time step Function
        # -- This will be used in the solver inner loop sub-stepping
        FS = FunctionSpace(particle_vom, "DG", 0)
        self._dt_fn = Function(FS, name="per-particle-time-step-function")

        self._fields = [self._dt_fn]

        # Record any expressions involving interpolates that are part of the update
        self._interp_exprs = []
        self._rebuild_interp = False


    # -- Define property methods
    @property
    def X(self):
        return self._X

    @property
    def dt_fn(self):
        return self._dt_fn

    @property
    def dt(self):
        return self._dt

    @property
    def dt_step(self):
        return self._dt_step

    @property
    def fields(self):
        """Returns a list of Functions owned by the stepper that must be eagerly migrated during a VOM rebuild."""
        return self._fields


    # NOTE: Using private attributes + property methods enforce a read-only contract on fields owned by the stepper.
    # Re-assigning the fields with a new Function (e.g., stepper.X = ...) would silently invalidate the update expr and associated cached parloops.
    # If we want to allow users to swap out the fields then this should be done via an explicit setter method:

    # @dt.setter
    # def dt(self, val):
    #     self._dt = val
    #     self.update_expr = self._build_update_expr()
    #     self.invalidate() # call invalidate to force parloops to be rebuilt

    def _sample_on_particles(self, field, tensor=False):
        FS = TensorFunctionSpace(self.particle_vom, "DG", 0) if tensor else VectorFunctionSpace(self.particle_vom, "DG", 0)
        return interpolate(field, FS)

    def _init_step_interp(self):
        self._step_interp = interpolate(self._update_expr, self._X.function_space())
        self._interp_exprs.append(self._step_interp)


    @abstractmethod
    def _build_update_expr(self):
        """Build update expressions for each stage."""
        pass
    

    def _reset_interpolators(self):
        """Clear cached Interpolators so they get rebuilt against the current VOM topology.
        Note that the symbolics of the underlying expressions remain valid."""
        for expr in [self._update_expr, *self._interp_exprs]:
            for node in extract_type(expr, Interpolate):
                node.__dict__.pop("_interpolator", None)


    def _check_interp(self):
        """Check if interpolators need to be rebuilt"""
        if self._rebuild_interp:
            self._reset_interpolators()
            self._rebuild_interp = False


    def invalidate(self):
        """Mark all interpolators as stale -> need to be rebuilt"""
        self._rebuild_interp = True


    def begin_outer_step(self, dt_step=None):
        """Sample the physical fields used by the update at the particles' current positions.
        Called once per outer time step and used throughout the entire inner crossing loop.
        `dt_step` is the step actually being taken, which may be shorter than the user-assigned 
        time step `self.dt` - namely, when the final step is truncated at `t_end`.
        """
        self._dt_step = self._dt if dt_step is None else float(dt_step)



    def begin_substep(self):
        pass


    def finalize_outer_step(self):
        pass


    def step(self):
        res = assemble(self._step_interp)
        return res


class ForwardEulerStepper(ParticleTimeStepper):

    def __init__(self, particle_vom, dt, v):
        super().__init__(particle_vom, dt)

        # Validate inputs
        if not isinstance(v, Expr):
            raise TypeError("Expected velocity to be supplied as a UFL expression.")
        
        if not isinstance(dt, float):
            raise TypeError("Expected dt to be supplied as a float")
        
        VFS = VectorFunctionSpace(particle_vom, "DG", 0)
        TFS = TensorFunctionSpace(particle_vom, "DG", 0)
        
        # Check if velocity is defined on the VOM or on the parent mesh
        if extract_unique_domain(v) is self.particle_vom:
            self._v_fn = v
            self._v_interp = None
        else:
            self._v_fn = Function(VFS, name="particle-v")
            self._v_interp = self._sample_on_particles(v)
    
        # Define quantities that should be refreshed per cell frame
        # -- invJ
        self._invJ_fn = Function(TFS, name="particle-invJ")
        self._invJ_interp = self._sample_on_particles(self._invJ_expr, tensor=True)

        # -- v_ref = invJ * v
        self._v_ref_fn = Function(VFS, name="particle-v-ref")
        self._v_ref_interp = interpolate(self._invJ_fn * self._v_fn, VFS)

        # Register interpolation expressions + fields
        self._interp_exprs.extend([self._invJ_interp, self._v_ref_interp])
        if self._v_interp is not None:
            self._interp_exprs.append(self._v_interp)
        self._fields.extend([self._v_fn, self._invJ_fn, self._v_ref_fn])

        self._build_update_expr()
        self._init_step_interp()

    def _build_update_expr(self):
        """X(t + dt) = X(t) + J^-1 * v * dt"""
        self._update_expr = self._X + self._dt_fn * self._v_ref_fn

    def begin_outer_step(self, dt_step=None):
        super().begin_outer_step(dt_step)
        self._check_interp()
        if self._v_interp is not None:
            assemble(self._v_interp, tensor = self._v_fn)

    def begin_substep(self):
        self._check_interp()
        assemble(self._invJ_interp, tensor=self._invJ_fn) # cross mesh -> current cell
        assemble(self._v_ref_interp, tensor=self._v_ref_fn) # pointwise on VOM (cheap)

    @property
    def v(self):
        return self._v_fn


    @property
    def v_ref(self):
        return self._v_ref_fn


class StormerVerletStepper(ParticleTimeStepper):

    def __init__(self, particle_vom, dt, a):
        super().__init__(particle_vom, dt)
        
        # Validation
        if not isinstance(a, Expr):
            raise TypeError("Expected acceleration to be supplied as a UFL expression.")
        
        self._a = a # physical acceleration -grad(phi)

        VFS = VectorFunctionSpace(particle_vom, "DG", 0)
        TFS = TensorFunctionSpace(particle_vom, "DG", 0)

        # Record acceleration at both the start and end of a time step
        self._a_start_fn = Function(VFS, name="particle-accel-start")
        self._a_end_fn = Function(VFS, name="particle-accel-end")
        self._a_interp = interpolate(self._a, VFS)

        # Record velocity
        self._v_fn = Function(VFS, name="particle-v") # evolving state
        self._v_half_fn = Function(VFS, name="particle-v-half") # v^{n+1/2}
        
        # Define quantities that should be refreshed per cell frame
        # -- invJ
        self._invJ_fn = Function(TFS, name="particle-invJ")
        self._invJ_interp = self._sample_on_particles(self._invJ_expr, tensor=True)

        # -- v_half_ref = invJ * v_half
        self._v_half_ref_fn = Function(VFS, name="particle-v-half-ref")
        self._v_half_ref_interp = interpolate(self._invJ_fn * self._v_half_fn, VFS)

        # Register interpolation expressions + fields
        self._interp_exprs.extend([self._a_interp, self._invJ_interp, self._v_half_ref_interp])
        self._fields.extend([self._a_start_fn, self._a_end_fn, self._v_fn, self._v_half_fn, self._v_half_ref_fn, self._invJ_fn])        

        self._build_update_expr()
        self._init_step_interp()


    def _build_update_expr(self):
        """X(t + dt) = X(t) + dt * J^-1 * (v + dt/2 * a_start)"""
        self._update_expr = self._X + self._dt_fn * self._v_half_ref_fn # compute X^{n+1} using V_ref^{n+1/2}
    
    def begin_outer_step(self, dt_step=None):
        super().begin_outer_step(dt_step)
        self._check_interp()
        assemble(self._a_interp, tensor=self._a_start_fn)
        self._v_half_fn.assign(self._v_fn + 0.5 * self._dt_step * self._a_start_fn)


    def begin_substep(self):
        self._check_interp()
        assemble(self._invJ_interp, tensor=self._invJ_fn) # cross-mesh -> current cell
        assemble(self._v_half_ref_interp, tensor=self._v_half_ref_fn) # pointwise on VOM


    def finalize_outer_step(self):
        assemble(self._a_interp, tensor=self._a_end_fn)                   
        self._v_fn.assign(self._v_fn + 0.5*self._dt_step*(self._a_start_fn + self._a_end_fn))

    
    @property
    def v(self):
        return self._v_fn

    
    @property
    def v_ref(self):
        return self._v_half_ref_fn

    
    





    





        