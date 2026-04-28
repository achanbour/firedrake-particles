from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from exceptions import BisectionNotConvergedError


class CellCrossingSolver(ABC):
    """
    Abstract base class for solvers used to detect particle collisions with mesh entities.
    """
    
    @abstractmethod
    def solve(self, stepper, ref_cell, particle_idxs, particle_dt):
        """Solves for the time at which each particle exited its current cell as well as
        for its position at the crossing point."""
        pass


@dataclass
class BisectionSolverParams:
    max_iters: int


BISECTION_COUNT = 0
class BisectionSolver(CellCrossingSolver):
    def __init__(self, params: BisectionSolverParams):
        self._params = params

    
    def solve(self, stepper, ref_cell, crossing_particle_idxs, crossing_particle_dt, bary_tol, time_tol):
        global BISECTION_COUNT
        BISECTION_COUNT += 1
        num_particles = len(crossing_particle_idxs)
        
        # Initialise the bisection bracket for each particle
        t_lo = np.zeros(num_particles, dtype=float)
        t_hi = crossing_particle_dt.copy()
        
        for _ in range(self._params.max_iters):
            t_mid = (t_lo + t_hi) / 2
            
            # Advance particles by the midpoint of their current bisection bracket
            stepper.dt_fn.dat.zero()
            stepper.dt_fn.data_wo[crossing_particle_idxs] = t_mid
            
            particle_pos_mid = stepper.step()
            X_mid = particle_pos_mid.dat.data[crossing_particle_idxs]
            bary_mid = ref_cell.compute_barycentric_coordinates(X_mid)

            # Evaluate the boolean predicate
            # For particles inside at the midpoint: advance the lower end of the bracket
            # For particles outside at the midpoint: advance the higher end of the bracket
            inside = np.all(bary_mid >= -bary_tol, axis=1)
            t_lo[inside] = t_mid[inside]
            t_hi[~inside] = t_mid[~inside]

            # Declare convergence if all brackets have sufficiently shrunk
            if np.max(t_hi - t_lo) < time_tol:
                break
        else:
            raise BisectionNotConvergedError(
                f"Bisection failed to converge within {self._params.max_iters} iterations using time_tol={time_tol}"
            )
        
        # Extract crossing times
        # In bisection terms, this corresponds to the last time the the boolean predicate was True (i.e., the particle was inside)
        t_cross = t_lo

        # Compute the position of each particle at the crossing point
        stepper.dt_fn.dat.zero()
        stepper.dt_fn.data_wo[crossing_particle_idxs] = t_cross
        particle_pos_cross = stepper.step()
        X_cross = particle_pos_cross.dat.data_ro[crossing_particle_idxs]
        bary_cross = ref_cell.compute_barycentric_coordinates(X_cross)

        return t_cross, X_cross, bary_cross
