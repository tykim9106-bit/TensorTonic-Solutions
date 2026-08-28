import numpy as np

def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    data = np.float64(left["data"]) + np.float64(right["data"])

    return {
        "id": output_id,
        "data": np.float64(data),
        "grad": np.float64(0.0),
        "op": "+",
        "parents": [left, right],
    }