import numpy as np

T = np.asarray([[1, 2], [3, 4]])

breakpoint()

"""
Python rewrites object[a, b, c] into a call to object.__getitem__((a,b,c))
(assuming object implements the __getitem__ method)

list.__getitem__(i) accepts i as list or slice. It doesn't implement a rule for Ellipsis.

NumPy arrays override __getitem__ and explicitly support tuple indexing (logic lives in NumPy's C implementation)
arr[..., 0] -> arr.__getitem__(Ellipsis, 0)
"""
T_slice = T[..., 0]

