
class IncompatibleMeshError(Exception):
    """Raised when user-supplied fields are defined on the wrong mesh(es)."""
    pass

class ParticleCrossingLoopNotConverged(RuntimeError):
    """Raised when particles could not complete their dt within the maximum number 
    of allowed iterations."""
    pass

class BisectionNotConvergedError(RuntimeError):
    """Raised when the bisection algorithm has not converged within the maximum number of allowed iterations"""
    pass
