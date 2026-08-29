import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    output = {}
    
    for key, train in train_dist.items():
        serving = serving_dist[key] 
        train = np.asarray(train, dtype = float) + eps
        serving = np.asarray(serving , dtype = float) + eps

        PSI = np.sum((serving - train) * np.log(serving/train))
        if PSI >= threshold:
            skewed = True
        else:
            skewed = False

        output[key] = {"psi": PSI, "skewed" : skewed}

    return output