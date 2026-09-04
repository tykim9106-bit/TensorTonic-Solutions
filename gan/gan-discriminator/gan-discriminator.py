import numpy as np

def discriminator(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Returns discriminator probabilities as a float64 array with shape (B, 1).
    """
    def sigmoid(a):
        return 1 / (1+np.exp(-a))

    return sigmoid(x @ W)