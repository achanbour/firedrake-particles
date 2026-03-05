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
        m = V.mesh()
        assert invJ.function_space().mesh() == m
        assert v.function_space().mesh() == m
        assert dt.function_space().mesh() == m
        
        # Euler update expression (in ref. space)
        self.update_expr = X + invJ * v * dt

        # Build the Interpolator object once (it caches the symbolic structure)
        self.interpolator = firedrake.Interpolator(
            self.update_expr,
            V,
            freeze_expr=False
        )

        # Since the update expression has no arguments, assemble computes the interpolation action
        # which is a Function
        self.output = firedrake.Function(V)

    def evaluate(self):
        # Reuse existing kernels and parloop
        """
        The symbolic structure of our update expression never changes, only the data values change:
        X.dat, invJ.dat, v.dat etc.
        
        Firedrake kernels read these values directly from the Dats each time the parloop runs.
        """

        return self.interpolator.assemble(tensor=self.output)
    
