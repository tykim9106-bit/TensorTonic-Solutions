import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    rms = torch.mean(x**2, axis = -1, keepdims = True)

    return x/torch.sqrt(rms + epsilon) * g
    