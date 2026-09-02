import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    B, S_q, D= q.shape
    head_dim = D // num_heads
    H = num_heads

    q = q.reshape(B, S_q, H, -1).transpose(1,2)
    k = k.reshape(B, -1, H, head_dim).transpose(1,2)

    scores = q @ k.transpose(-1,-2)
    scores = scores/torch.sqrt(torch.tensor(head_dim, dtype = q.dtype))

    return scores

    
