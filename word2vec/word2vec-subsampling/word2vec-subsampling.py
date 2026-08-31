import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    total = counts.sum()
    
    f = counts/total

    keep_probs = torch.minimum(
        torch.tensor(1.0),
        torch.sqrt(t/f) 
    )

    return keep_probs
