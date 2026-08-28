import numpy as np

def value_multiplication_node(left, right, output_id):
    """
    Returns: a multiplication node that retains the two supplied leaf records as ordered parents
    """
    data = np.float64(left["data"]) * np.float64(right["data"])

    return {
        "id" : output_id,
        "data" : (data),
        "grad" : np.float64(0.0),
        "op" : "*",
        "parents" : [left, right],
    }
