import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x, dtype = float)
    m = np.max(x, axis = -1, keepdims=True)

    x = x-m
    return np.exp(x)/np.sum(np.exp(x), axis = -1, keepdims=True)
    