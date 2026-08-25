import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    l_i = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return float(np.mean(l_i))
    else:
        return float(np.sum(l_i))