import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis = -1, keepdims = True)
    var = np.var(x, axis = -1, keepdims = True)

    return gamma* (x-mean)/np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    head_dim = Q.shape[-1] // num_heads
    q_heads = Q @ W_q #B, seq_len, 
    k_heads = K @ W_k
    v_heads = V @ W_v

    B, seq_len, _ = q_heads.shape

    q_heads = q_heads.reshape(B, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    k_heads = k_heads.reshape(B, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    v_heads = v_heads.reshape(B, seq_len, num_heads, head_dim).transpose(0,2,1,3)

    score = q_heads @ k_heads.transpose(0,1,3,2)#(B, H, seq_q, seq_k)
    score = score/np.sqrt(head_dim)
    attention_score = softmax(score, axis =-1) #seq_k

    heads = attention_score @ v_heads #(B, H, seq_q, head_dim)
    head = heads.transpose(0,2,1,3).reshape(B, seq_len, -1)

    return head @ W_o
    
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    h = x @ W1 + b1
    h = np.maximum(0, h)

    return h @ W2 + b2
    

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    attn_out = multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads)
    x_prime = layer_norm(x + attn_out, gamma1, beta1)

    ffn_out = feed_forward(x_prime, W1, b1, W2, b2)
    out = layer_norm(x_prime + ffn_out, gamma2, beta2)

    return out