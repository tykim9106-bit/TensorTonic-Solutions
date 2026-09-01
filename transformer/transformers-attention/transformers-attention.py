import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    batch, seq_len_q, d_k = Q.shape

    score = Q @ K.transpose(-1,-2)
    score = score / math.sqrt(d_k)
    W = F.softmax(score, dim = -1)

    return W @ V