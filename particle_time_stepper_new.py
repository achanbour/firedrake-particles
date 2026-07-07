from abc import ABC, abstractmethod
from functools import partial
from collections import namedtuple

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


Stage = namedtuple("Stage", ["target", "expr", "relocate"])
# target : Function the stage writes into (X or V)
# expr   : symbolic UFL update, interpolated onto target.function_space()
# relocate : re-locate particles in the VOM after this stage (needed when a
#            later stage samples a parent-mesh field at the updated positions)

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
        
        # Register fields that need to be eagerly rebuilt when the particle VOM changes
        # self._fields = [self._dt_fn]

        self._stages = []
        self._stage_callables = []
        self._step_callable_is_current = False


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
        # F = Function(FS)
        # XXX: F only exist to ensure the FS gets appropriately rebuilt when the VOM changes
        # self._fields.append(F)
        return interpolate(field, FS)


    def _v_pullback(self, v):
        invJ = self._invJ_expr
        if extract_unique_domain(v) is self.particle_vom:
            return self._sample_on_particles(invJ, tensor=True) * v
        else:
            return self._sample_on_particles(invJ * v)


    # -- Define shared core methods
    @abstractmethod
    def _build_update_expr(self):
        """Build update expressions for each stage."""
        pass


    def _build_step_callable(self):
        self._stage_callables = [
            (s.target,
             partial(assemble, interpolate(s.expr, self._X.function_space())),
             s.relocate
            )
            for s in self._stages
        ]
        self._step_callable_is_current = True
    

    def _check_step_callable_is_current(self):
        """Rebuild the step callable."""
        if not self._step_callable_is_current:
            self._build_step_callable()
    
    # def _rebuild_fields(self):
    #     """Rebuild all the fields owned by the stepper after a VOM mutation"""
    #     for f in self._fields:
    #         if isinstance(f, Function):
    #             f._match_mesh_topology_version()
    #         else:
    #             raise TypeError("Cannot rebuild a field that is not a Function.")


    def invalidate(self):
        """Mark all callables as stale."""
        # self._rebuild_fields()
        self._step_callable_is_current = False


    def begin_substep(self):
        pass


    def finalize_substep(self):
        pass


    def step(self):
        self._check_step_callable_is_current()
        result = None
        for target, call, relocate in self._stage_callables:
            result = call()
            if target is not None:
                # Record states that are not X
                target.assign(result)
            if relocate:
                # TODO
                # Write the updated positions into the VOM so that next stage can evaluate the required fields
                # at the new particle positions
                pass
        return result


class ForwardEulerStepper(ParticleTimeStepper):
    """
    Advance particles using the Forward Euler update.
    """

    def __init__(self, particle_vom, dt, v):
        super().__init__(particle_vom, dt)

        if not isinstance(v, Expr):
            raise TypeError("Expected velocity to be supplied as a UFL expression.")
        
        self._v = v    
        # if extract_unique_domain(v) is self.particle_vom:
        #     self._fields.append(self._v)

        self._v_ref = self._v_pullback(self._v)

        self._build_update_expr()
        self._build_step_callable()


    def _build_update_expr(self):
        """X(t + dt) = X(t) + J^-1 * v * dt"""
        self._stages = [
            Stage(None, self._X + self._dt_fn * self._v_ref, relocate=False)
        ]
    
    @property
    def v(self):
        return self._v

    @property
    def v_ref(self):
        return assemble(interpolate(self._v_ref, self.particle_vom.coordinates.function_space()))
    
    
class StormerVerletStepper(ParticleTimeStepper):
    """Advance particles using the 3 stage Stormer-Verlet update."""

    def __init__(self, particle_vom, dt, a):
        super().__init__(particle_vom, dt)
    
        if not isinstance(a, Expr):
            raise TypeError("Expected acceleration to be supplied as a UFL expression.")
        
        self._a = a # physical acceleration -grad(phi)

        # -- record acceleration at both the start and end of a step (needed for the velocity update)
        self._a_start = Function(
            VectorFunctionSpace(particle_vom, "DG", 0),
            name="particle-acceleration-start"
        )
        self._a_end = Function(
            VectorFunctionSpace(particle_vom, "DG", 0),
            name="particle-acceleration-end"
        )

        # -- velocity is an evolving state
        self._v = Function(
            VectorFunctionSpace(particle_vom, "DG", 0), 
            name="particle-velocity"
        )

        # self._fields.append(self._a_start)
        # self._fields.append(self._a_end)
        # self._fields.append(self._v)

        # -- for position update
        self._v_half = self._v + 0.5 * self._dt_fn * self._a_start
        self._v_half_ref = self._v_pullback(self._v_half)

        self._v_update = self._v + 0.5 * self._dt_fn * (self._a_start + self._a_end)

        # XXX: has to be part of step callables?
        self._v_update_callable = partial(
            assemble,
            interpolate(self._v_update, self._v.function_space())
        )

        self._build_update_expr()
        self._build_step_callable()

    # XXX: Updates position only
    def _build_update_expr(self):
        """X(t + dt) = X(t) + dt * J^-1 * (v + dt/2 * a_start)"""
        self._stages = [
            Stage(None, self._X + self._dt_fn * self._v_half_ref, relocate=False) # X^{n+1} using V_ref^{n+1/2}
        ]
    
    def begin_substep(self):
        # Cache a(x_n) when VOM has current state x_n
        self._a_start.assign(
            assemble(interpolate(self._a, self._a_start.function_space()))
        )
    
    def finalize_substep(self):
        # Called after update_ref_view has committed the new positions into the VOM
        # -- Do a full velocity update
        self._a_end.assign(
            assemble(interpolate(self._a, self._a_end.function_space()))
        )                    
        self._v.assign(self._v_update_callable())
    
    @property
    def v(self):
        return self._v
    
    @property
    def v_ref(self):
        # return the reference space velocity used for the position drift step
        return assemble(interpolate(self._v_half_ref, self.particle_vom.coordinates.function_space()))


    
    





    





        