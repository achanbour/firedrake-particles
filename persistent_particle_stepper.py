import firedrake

class EulerParticleStepper:
    def __init__(self, X, invJ, v, dt):
        self.X = X
        self.invJ = invJ
        self.v = v
        self.dt = dt

        V = X.function_space()

        # All args. assumed to be functions
        # so we check they're all defined on the same mesh
        assert invJ.function_space().mesh() == V
        assert v.function_space().mesh() == V
        assert dt.function_space().mesh() == V
        
        # Euler update expression (in ref. space)
        self.update_expr = X + invJ * v * dt

        # Build the Interpolator object once (it caches the symbolic structure)
        self.interpolator = firedrake.Interpolator(
            self.update_expr,
            V
        )

        # Since the update expression has no arguments, assemble computes the interpolation action
        # which is a Function
        self.output = firedrake.Function(V)

    def evaluate(self):
        # Reuse existing parloop
        return self.interpolator.assemble(tensor=self.output)
    
