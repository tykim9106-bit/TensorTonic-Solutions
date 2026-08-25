import math
import numpy as np

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x)
    y = np.vectorize(math.erf)(x/math.sqrt(2))

    return x*(1+y)/2
    