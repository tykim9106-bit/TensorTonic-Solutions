import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    pairs =[]
    n = len(token_ids)

    for i in range(n):
        start = max(0, i-window)
        end = min(n, i+window+1)

        for j in range(start, end):
            if i == j:
                continue

            pairs.append([token_ids[i].item(), token_ids[j].item()])

    return torch.tensor(pairs, dtype = int).reshape(-1, 2)
