import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    head_dim = Q.shape[-1] // num_heads

    q_heads = Q @ W_q #B, seq_q, H*head_dim
    k_heads = K @ W_k
    v_heads = V @ W_v

    q_heads = q_heads.reshape(Q.shape[0], Q.shape[1], num_heads, head_dim)
    k_heads = k_heads.reshape(K.shape[0], K.shape[1], num_heads, head_dim)
    v_heads = v_heads.reshape(V.shape[0], V.shape[1], num_heads, head_dim)
    
    q_heads = q_heads.transpose(0, 2, 1, 3) #batch, num_head, seq_q, head_dim
    k_heads = k_heads.transpose(0, 2, 1, 3)
    v_heads = v_heads.transpose(0, 2, 1, 3) #B, H, seq_v, head_dim

    scores = q_heads @ k_heads.transpose(0, 1, 3, 2) #B, H, head_dim, s_k
    scaled_scores = scores / np.sqrt(head_dim) #B, H, seq_q, seq_k

    attention_weights = softmax(scaled_scores, axis=-1) #seq_k, B, H, seq_q, seq_k
    head_outputs = attention_weights @ v_heads #B, H, seq_q, head_dim

    head_outputs = head_outputs.transpose(0, 2, 1, 3) #B, seq_q, H, head_dim
    head = head_outputs.reshape(Q.shape[0], Q.shape[1], Q.shape[-1]) #B, seq_q, H*head_dIm

    return head @ W_o

    