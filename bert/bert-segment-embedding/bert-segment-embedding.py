import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    # token_ids: (B, S)
    # token_embeddings: (V, H)
    # position_embeddings: (S, H)
    # segment_embeddings: (2, H)

    token_emb = token_embeddings[token_ids]       # (B, S, H)
    pos_emb = position_embeddings[:token_ids.shape[1]]  # (S, H)
    seg_emb = segment_embeddings[segment_ids]     # (B, S, H)

    return (token_emb + pos_emb + seg_emb).astype(np.float64)