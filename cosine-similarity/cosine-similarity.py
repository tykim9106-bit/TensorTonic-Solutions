import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)

    dot = a.dot(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    result = 0.0

    if norm_a !=0 and norm_b !=0:
        result=float(dot/(norm_a * norm_b))

    return result