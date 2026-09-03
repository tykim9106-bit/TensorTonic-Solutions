import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    def layer_norm(x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    
        return (x - mean) / np.sqrt(var + eps)

    def gelu(x):
        return 0.5 * x * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
            )
        )

    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    B, N, D = x.shape
    head_dim = D // num_heads
    hidden_dim = int(D * mlp_ratio)

    # Weight 초기화
    if Wq is None:
        Wq = np.random.randn(D, D) * 0.02
    if Wk is None:
        Wk = np.random.randn(D, D) * 0.02
    if Wv is None:
        Wv = np.random.randn(D, D) * 0.02
    if Wo is None:
        Wo = np.random.randn(D, D) * 0.02

    if W1 is None:
        W1 = np.random.randn(D, hidden_dim) * 0.02
    if W2 is None:
        W2 = np.random.randn(hidden_dim, D) * 0.02

    x_norm = layer_norm(x)
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)

    scores = softmax(Q @ K.transpose(0,1,3,2) / np.sqrt(head_dim))
    attn = scores @ V 
    attn = attn.transpose(0,2,1,3).reshape(B, N, D)
    attn_out = attn @ Wo
    
    x = x + attn_out
    x_norm = layer_norm(x)
    hidden = x_norm @ W1
    hidden = gelu(hidden)
    mlp_out = hidden @ W2

    x = x + mlp_out

    return x
    
    