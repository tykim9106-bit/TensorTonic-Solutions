import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reconstruction with shape (B, D).
    """
    def sigmoid(s):
        return 1 / (np.exp(-s) + 1)

    hidden = z @ W_dec + b_dec
    return sigmoid(hidden)