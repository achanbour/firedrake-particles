
class ParticleCrossingLoopNotConverged(RuntimeError):
    """Raised when particles could not complete their dt within the maximum number 
    of allowed iterations."""
    pass

class BisectionNotConvergedError(RuntimeError):
    """Raised when the bisection algorithm has not converged within the maximum number of allowed iterations"""
    pass

class NoParticlesRemainingError(RuntimeError):
    """Raised when no particles remain in the simulation."""