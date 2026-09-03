import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns the batch-mean KL divergence as a Python float.
    """
    kl = 1 + log_var - mu ** 2 - np.exp(log_var)
    kl = np.sum(kl, axis = -1, keepdims =False)

    return -1 / 2 * float(np.mean(kl))