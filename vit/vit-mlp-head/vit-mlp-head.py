import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    def layernorm(X, eps = 10**(-6)):
        mean = np.mean(X , axis = -1, keepdims = True)
        var = np.var(X, axis = -1, keepdims = True)

        return (X - mean)/np.sqrt(var + eps)
    
    B, N, D = encoder_output.shape
    cls = encoder_output[:,0,:]

    logits = layernorm(cls) @ W_head
    return np.float64(logits)
    