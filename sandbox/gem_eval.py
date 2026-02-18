import numpy as np

def evaluate_gem(gem_expr, rt_point):
    """Evaluate a rank-1 GEM tensor expression at a single runtime point"""
    import gem
    import gem.impero_utils as impero_utils
    from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder
    
    n, = gem_expr. shape

    # Free index over components
    k = gem.Index("k", extent=n)

    # Output GEM varibale A[k]
    A = gem.Variable("A", shape=(n, ))
    return_expr = gem.Indexed(A, (k, ))

    # Expression component: expr[k]
    evaluation_expr = gem.Indexed(gem_expr, (k, ))

    # Lower GEM -> Impero (loop IR)
    # Express rank-1 assignment as an Impero program involving loops (scheduled tensor algebra)
    impero_c = impero_utils.compile_gem([(return_expr, evaluation_expr)], (k, ))

    # Build a TSFC ExpressionKernel
    kernel_builder = ExpressionKernelBuilder("double")

    # Collect kernel arguments
    # No coefficients/constants: only runtime coords are required
    kernel_builder.set_coefficient_numbers(())
    kernel_builder.set_coefficients([])
    kernel_builder.set_constants([]) 

    # Output variable is a kernel argument
    kernel_builder.set_output(A)

    # Infer other kernel arguments from the dependencies of the GEM evaluation expression tree (e.g., rt_X)
    kernel_builder.register_requirements([evaluation_expr])

    # Build the kernel (wraps a Loopy TranslationUnit)
    kernel = kernel_builder.construct_kernel(impero_c, {}, False, False)

    # import loopy 
    # ccode = loopy.generate_code_v2(kernel.ast) # string of generated C code 

    """
    # C device code is the arithmetic code 
    void expression_kernel(double* A, double const* rt_X)
    {
    t0[0] = 1 - rt_X[0] - rt_X[1];
    t0[1] = rt_X[0];
    t0[2] = rt_X[1];
    for i:
        A[i] += t0[i];
    }
    """

    # Execute the kernel via PyOP2
    from pyop2 import op2
    from pyop2.local_kernel import LoopyLocalKernel

    tu = kernel.ast # Loopy TranslationUnit (kernel IR)
    lk = LoopyLocalKernel(tu, "expression_kernel") # runtime wrapper of the kernel IR (specifies the entrypoint kernel)

    # Define an iteration set of size 1 as we only have one point
    iterset = op2.Set(1)

    # Provide a concrete runtime input point
    rt_point = np.asarray(rt_point)
    rt_X_global = op2.Global(rt_point.size, data=rt_point)

    # Allocate output buffer
    A_out = np.zeros(3, dtype=float)
    A_global = op2.Global(3, data=A_out)
    
    # Execute the kernel
    # JIT compiles the executable kernel into a shared library
    # loading this shared library produces a pointer to a callable C function
    # this function then gets called in a par_loop (only once here)
    op2.par_loop(
        lk,
        iterset,
        A_global(op2.INC), # increment access (kernel computes increments to be summed into a global output object)
        rt_X_global(op2.READ), # read-only access
    )
    return A_out
