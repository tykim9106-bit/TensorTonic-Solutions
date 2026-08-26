import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x, dtype = float)
    gamma = np.asarray(gamma, dtype = float)
    beta = np.asarray(beta, dtype = float)
    
    if x.ndim == 2:
        mean = np.mean(x, axis = 0, keepdims = True)
        var = np.var(x, axis = 0, keepdims = True)
    elif x.ndim == 4:
        mean = np.mean(x, axis = (0, 2, 3), keepdims = True)
        var = np.var(x, axis = (0, 2, 3), keepdims = True)
        gamma = np.reshape(gamma, (1,-1,1,1))
        beta = np.reshape(beta, (1,-1,1,1))
        

    x_hat = (x-mean)/np.sqrt(var + eps)
    y = gamma*x_hat + beta

    return y

    