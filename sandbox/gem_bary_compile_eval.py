import numpy as np

def evaluate_gem(gem_expr, rt_point):
    """Evaluate a rank-1 GEM tensor expression at a single runtime point."""
    import gem
    import gem.impero_utils as impero_utils
    from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder
    from pyop2.mpi import COMM_WORLD

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

    # Build the kernel
    kernel = kernel_builder.construct_kernel(impero_c, {}, False, False)
    
    # The TSFC ExpressionKernel wraps a Loopy TranslationUnit (kernel IR)
    # It is the ultimate input (after all the pre-processing is done) to the compiler that generates a single object file
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

    tu = kernel.ast

    # LoopyLocalKernel is a a PyOP2 structure
    # It attaches PyOP2 concepts (defined globally) to the Loopy kernel (operating locally)
    lk = LoopyLocalKernel(tu, "expression_kernel")

    # Define an iteration set of size 1 as we only have one point
    iterset = op2.Set(1)

    # Provide a concrete runtime input point
    # NOTE: op2.Global is a flat vector of length dim so dim must be equal to the number of scalar entries in data
    # NOTE: in this simple case, global and local are isomorphic so we can define op2.Dat with an identity mapping instead of op2.Global variables
    rt_point = np.asarray(rt_point)
    rt_X_global = op2.Global(rt_point.size, data=rt_point, comm=COMM_WORLD)

    # Allocate output buffer
    A_out = np.zeros(n, dtype=float)
    A_global = op2.Global(n, data=A_out, comm=COMM_WORLD)
    
    # Execute the kernel in a PyOP2 par_loop (parallel_loop)
    op2.par_loop(
        lk,
        iterset,
        A_global(op2.INC), # increment access (kernel computes increments to be summed into a global output object)
        rt_X_global(op2.READ), # read-only access
    )

    # NOTE: The execution steps are:
    # PyOP2 JIT compiles the executable kernel into a shared library,
    # then loading this shared library in Python produces a pointer to a callable C function,
    # this function then gets called in a par_loop (only once here) with pointers to the global arrays/Dats as arguments

    # NOTE: When assembling operators of forms, the kernel arguments are op2.Dats with op2.Maps mapping op2.Sets of function DoFs 
    # to the op2.Sets of mesh topological entities (cells, edges etc. on which Dofs are defined)
    # This is because DoFs are cell-local while topological entities have both a local and global numbering
    # and the latter is used to construct the global operator

    return A_out

"""
# rt_point.point_size gives the total number of scalar entries

# Single point vector
rt_point = np.array([0.25, 0.25])
rt_point.shape == (2,)
rt_point.size == 2

# Single point in a tensor storing a batch of points
rt_point = np.array([[0.25, 0.25]])
rt_point.shape == (1,2)
rt_point.size == 2

# Multiple points in a tensor storing a batch of points
rt_point = np.array([[0.1,0.2],
                     [0.3,0.4]])
rt_point.shape == (2,2)
rt_point.size == 4
"""