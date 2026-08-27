import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x = np.asarray(x, dtype = float)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    
    def relu(x):
        return np.maximum(0, x)

    h = relu(x @ W1.T)
    y = relu(h @ W2.T +x)

    return y
