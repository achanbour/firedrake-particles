import numpy as np
import gem
import gem.impero_utils as impero_utils
from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder
from tsfc.loopy import generate
import loopy
from pyop2 import op2
from pyop2.mpi import COMM_WORLD
from pyop2.local_kernel import LoopyLocalKernel

"""
Construct a GEM expression for the matrix multiplication of two GEM tensors 
then build and compile the evaluation kernel.
"""

# Build the matmul expression
A = gem.Variable("A", shape=(3, 2))
B = gem.Variable("B", shape=(2, 4)) 
expr = A @ B

# Define return expression
C = gem.Variable("C", shape=(3, 4)) # has two free indices
i = gem.Index("i", extent=3)
j = gem.Index("j", extent=4)

# Lower GEM (tensor algebra) -> Impero (loop DAG)
# Produces nested looops over free indices i and j and at the inner-most level
# over the contraction index k (of extent=shared dim. = 2)
impero_c = impero_utils.compile_gem(
    [(gem.Indexed(C, (i, j)), gem.Indexed(expr, (i, j)))], 
    (i, j)
)

"""
# Build a TSFC expression kernel
kernel_builder = ExpressionKernelBuilder(scalar_type="double")

# Collect kernel arguments
# NOTE: We cannot pass GEM variables into set_coefficients of TSFC.ExpressionKernelBuilder
# as it expects UFL Coefficients. We do need a way to pass the input matrices as arguments to our kernel.

kernel_builder.set_coefficient_numbers((0, 1))
kernel_builder.set_coefficients([A, B])
kernel_builder.set_constants([])

# Output accumulation tensor is a kernel argument
kernel_builder.set_output(C)

# Register dependencies of the evaluation expression as kernel arguments
kernel_builder.register_requirements([expr])

# Build the kernel
kernel = kernel_builder.construct_kernel(
    impero_c, 
    index_names={}, 
    needs_external_coords=False, 
    log=False
)
"""
# TSFC's ExpressionKernel wraps a Loopy TranslationUnit (kernel IR)
# from which the compiler (in this case loopy) then generates a single SO
# Hence, we can directly generate the C code of the kernel we just built
# by converting this IR into C code

# Bypass TSFC ExpressionKernelBuilder and build a kernel directly in loopy

# Build the argument set manually
gem_vars = gem.extract_type((expr,), gem.Variable)

loopy_args = []
for var in gem_vars:
    loopy_args.append(
        loopy.GlobalArg(var.name,
                        dtype=np.float64,
                        shape=var.shape)
    )

loopy_args.insert(0,
    loopy.GlobalArg(C.name, dtype=np.float64, shape=C.shape)
)

kernel, event = generate(
    impero_c,
    loopy_args,
    np.float64,
    "my_kernel",
    index_names={}
)

# Extract the generated code
cgr = loopy.generate_code_v2(kernel) # TranslationUnitCodeGenerationResult
c_code = cgr.device_code()

# Execute the kernel via PyOP2

# Prepare runtime input
A_np = np.random.randn(3, 2)
B_np = np.random.randn(2, 4)
C_np = np.zeros((3, 4))
expected = A_np @ B_np

# Wrap input tensors as PyOP2 Globals
A_glob = op2.Global(A_np.size, data=A_np, comm=COMM_WORLD)
B_glob = op2.Global(B_np.size, data=B_np, comm=COMM_WORLD)
C_glob = op2.Global(C_np.size, data=C_np, comm=COMM_WORLD)

lk = LoopyLocalKernel(kernel, "my_kernel")
iterset = op2.Set(1)
op2.par_loop(
    lk,
    iterset,
    C_glob(op2.INC),
    A_glob(op2.READ),
    B_glob(op2.READ),
)
print(C_glob.data)
print(np.allclose(C_glob.data.reshape(3, 4), expected))

breakpoint()










