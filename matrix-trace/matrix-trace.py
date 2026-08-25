import numpy as np

def matrix_trace(A):
    matrix = np.asarray(A)
    total = 0.0

    for index in range(matrix.shape[0]):
        total += matrix[index, index]

    return float(total)