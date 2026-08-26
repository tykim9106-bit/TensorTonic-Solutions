import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p)
    q = np.asarray(q)

    mask = p > 0

    p = p[mask]
    q = q[mask]

    q = np.maximum(q, eps)

    result = np.sum(p * np.log(p / q))

    return float(result)