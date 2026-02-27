import numpy as np
import gem
from gem.interpreter import evaluate

"""
Construct a GEM expression for the matrix multiplication of two GEM tensors 
then evaluate it using the GEM interpreter.
"""

# Build the matmul expression
A = gem.Variable("A", shape=(3, 2))
B = gem.Variable("B", shape=(2, 4)) 
expr = A @ B

# Define the runtime input
A_np = np.random.randn(3, 2)
B_np = np.random.randn(2, 4)
expected = A_np @ B_np

res = evaluate((expr,), bindings={A: A_np, B: B_np})

print(np.allclose(res[0].arr, expected))

breakpoint()










