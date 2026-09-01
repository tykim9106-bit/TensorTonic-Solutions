import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate the vanishing gradient problem.
    Returns a list of gradient norms.
    """
    spectral_norm = np.linalg.norm(W_hh, ord=2)

    grad_norm = 1.0
    result = []

    for _ in range(T):
        result.append(grad_norm)
        grad_norm *= spectral_norm

    return result