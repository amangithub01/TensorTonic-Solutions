import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    arr = np.array(A)
    n, m = arr.shape

    # Initialize a new array with swapped dimensions (M, N)
    transposed = np.zeros((m, n), dtype=arr.dtype)

    # Manually map (i, j) -> (j, i)
    for i in range(n):
        for j in range(m):
            transposed[j, i] = arr[i, j]

    return transposed
    # pass
