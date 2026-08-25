import math


def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    return [
        value if value>0 else alpha*(math.exp(value)-1)
        for value in x
        
    ]