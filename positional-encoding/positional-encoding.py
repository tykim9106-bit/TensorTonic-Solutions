import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    PE = np.zeros((seq_len, d_model))
    p = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]

    angle_rates = 1 / np.power(base, (2 * (i // 2)) / d_model)
    angles = p * angle_rates

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
    