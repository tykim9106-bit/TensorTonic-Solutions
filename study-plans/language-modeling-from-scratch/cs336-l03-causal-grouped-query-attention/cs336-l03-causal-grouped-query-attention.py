import torch
import math

def causal_gqa(q, k, v):
    """
    Returns: causal grouped-query attention tensor
    """
    B, H_q, S, D = q.shape
    _, H_kv, _, _ = k.shape

    G = H_q // H_kv

    q_grouped = q.reshape(B, H_kv, G, S, D)

    scores = torch.einsum( "bngtd,bnsd->bngts", q_grouped, k, )
    scores = scores / math.sqrt(D)

    causal_mask = torch.triu( torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1, )

    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)

    out = torch.einsum( "bngts,bnsd->bngtd", attn, v, )   
    out = out.reshape(B, H_q, S, D)

    return out