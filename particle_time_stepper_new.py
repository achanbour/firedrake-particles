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
        
        # Define the fields owned by the stepper
        self.particle_vom = particle_vom
        self._X = self.particle_vom.reference_coordinates
        self._dt = dt

        x = SpatialCoordinate(self.particle_vom._parent_mesh)
        self._invJ_expr = inv(ReferenceGrad(x)) # used for every position update

        # Define a per-particle time step Function
        # -- This will be used in the solver inner loop sub-stepping
        FS = FunctionSpace(particle_vom, "DG", 0)
        self._dt_fn = Function(FS, name="per-particle-time-step-function")

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


    # NOTE: Using private attributes + property methods enforce a read-only contract on fields owned by the stepper.
    # Re-assigning the fields with a new Function (e.g., stepper.X = ...) would silently invalidate the update expr and associated cached parloops.
    # If we want to allow users to swap out the fields then this should be done via an explicit setter method:

    # @dt.setter
    # def dt(self, val):
    #     self._dt = val
    #     self.update_expr = self._build_update_expr()
    #     self.invalidate() # call invalidate to force parloops to be rebuilt


    # -- Define shared helper methods
    def _sample_on_particles(self, field, tensor=False):
        FS = TensorFunctionSpace(self.particle_vom, "DG", 0) if tensor else VectorFunctionSpace(self.particle_vom, "DG", 0)
        return interpolate(field, FS)


    def _v_pullback(self, v):
        invJ = self._invJ_expr
        if extract_unique_domain(v) is self.particle_vom:
            return self._sample_on_particles(invJ, tensor=True) * v
        else:
            return self._sample_on_particles(invJ * v)


    # -- Define shared core methods
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


    def begin_step(self):
        pass


    def finalize_step(self):
        pass


    def step(self):
        res = assemble(self._step_interp)
        return res


class ForwardEulerStepper(ParticleTimeStepper):
    """
    Advance particles using the Forward Euler update.
    """

    def __init__(self, particle_vom, dt, v):
        super().__init__(particle_vom, dt)

        if not isinstance(v, Expr):
            raise TypeError("Expected velocity to be supplied as a UFL expression.")
        
        self._v = v
        self._v_ref = self._v_pullback(self._v)
        self._v_ref_fn = Function(
            VectorFunctionSpace(particle_vom, "DG", 0),
            name="particle-v-ref"
        )
        self._v_ref_interp = interpolate(self._v_ref, self._v_ref_fn.function_space())
        self._interp_exprs.append(self._v_ref_interp)

        self._build_update_expr()
        self._init_step_interp()

    def _build_update_expr(self):
        """X(t + dt) = X(t) + J^-1 * v * dt"""
        self._update_expr = self._X + self._dt_fn * self._v_ref_fn


    def begin_step(self):
        # check if interpolators have to be rebuilt
        self._check_interp()
        assemble(self._v_ref_interp, tensor=self._v_ref_fn)

    @property
    def v(self):
        v_fn = assemble(self._sample_on_particles(self._v))
        v_fn.rename("particle-velocity")
        return v_fn


    @property
    def v_ref(self):
        return self._v_ref_fn

    
class StormerVerletStepper(ParticleTimeStepper):
    """Advance particles using the 3 stage Stormer-Verlet update."""

    def __init__(self, particle_vom, dt, a):
        super().__init__(particle_vom, dt)
    
        if not isinstance(a, Expr):
            raise TypeError("Expected acceleration to be supplied as a UFL expression.")
        
        self._a = a # physical acceleration -grad(phi)

        # -- record acceleration at both the start and end of a step (needed for the velocity update)
        self._a_start_fn = Function(
            VectorFunctionSpace(particle_vom, "DG", 0),
            name="particle-acceleration-start"
        )

        self._a_end_fn = Function(
            VectorFunctionSpace(particle_vom, "DG", 0),
            name="particle-acceleration-end"
        )

        self._a_interp = interpolate(self._a, self._a_start_fn.function_space())
        self._interp_exprs.append(self._a_interp)

        # -- velocity is an evolving state -> Function on VOM
        self._v_fn = Function(
            VectorFunctionSpace(particle_vom, "DG", 0), 
            name="particle-velocity"
        )

        # -- half-kick of velocity used for the position update
        self._v_half = self._v_fn + 0.5 * self._dt_fn * self._a_start_fn

        # NOTE: In SV, velocity is always a particle state so the pullback only requires sampling invJ
        # self._v_half_ref = self._v_pullback(self._v_half)
        self._invJ_fn = Function(
            TensorFunctionSpace(particle_vom, "DG", 0),
            name="particle-invJ"
        )
        self._invJ_interp = self._sample_on_particles(self._invJ_expr, tensor=True)
        self._interp_exprs.append(self._invJ_interp)
        
        self._v_half_ref = self._invJ_fn * self._v_half
        self._v_half_ref_interp = interpolate(self._v_half_ref, VectorFunctionSpace(particle_vom, "DG", 0))
        self._interp_exprs.append(self._v_half_ref_interp)

        # -- full velocity update using a(x^n) and a(x^n+1)
        self._v_update = self._v_fn + 0.5 * self._dt_fn * (self._a_start_fn + self._a_end_fn)
        self._v_update_interp = interpolate(self._v_update, self._v_fn.function_space())
        self._interp_exprs.append(self._v_update_interp)

        self._build_update_expr()
        self._init_step_interp()


    def _build_update_expr(self):
        """X(t + dt) = X(t) + dt * J^-1 * (v + dt/2 * a_start)"""
        self._update_expr = self._X + self._dt_fn * self._v_half_ref # X^{n+1} using V_ref^{n+1/2}
    

    def begin_step(self):
        # check if interpolators need to be rebuilt
        self._check_interp()
        # Compute and cache a(x^n) when VOM has current state x^n
        assemble(self._a_interp, tensor=self._a_start_fn)
        assemble(self._invJ_interp, tensor=self._invJ_fn)


    def finalize_step(self):
        # Do a full velocity update after the crossing solver has resolved all the positions
        assemble(self._a_interp, tensor=self._a_end_fn)                   
        self._v_fn.assign(assemble(self._v_update_interp)) # target appears in expr

    
    @property
    def v(self):
        return self._v_fn

    
    @property
    def v_ref(self):
        return assemble(self._v_half_ref_interp)

    
    





    





        