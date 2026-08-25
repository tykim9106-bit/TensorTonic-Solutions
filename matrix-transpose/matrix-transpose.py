import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    output=[]

    for j in range(len(A[0])):
        row = []
        for i in range(len(A)):
            row.append(A[i][j])
        output.append(row)  

    return np.asarray(output)