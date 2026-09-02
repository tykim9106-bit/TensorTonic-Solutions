import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.asarray(scores, dtype=float)
    sequence_length = scores.shape[-1]
    future = np.triu(
        np.ones((sequence_length, sequence_length), dtype=bool),
        k=1,
    )
    masked = scores.copy()
    masked[..., future] = mask_value
    return masked