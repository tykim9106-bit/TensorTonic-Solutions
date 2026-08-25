import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype = float)
    return x*np.exp(-np.logaddexp(0.0,-x))
    