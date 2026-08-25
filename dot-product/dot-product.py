import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    return float(sum(a * b for a, b in zip(x, y)))