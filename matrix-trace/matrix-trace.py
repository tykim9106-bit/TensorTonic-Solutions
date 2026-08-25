import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    # Write code here
    n, n = np.asarray(A).shape
    sum = 0.0
    
    for i in range(n):
        sum += A[i][i]

    return sum