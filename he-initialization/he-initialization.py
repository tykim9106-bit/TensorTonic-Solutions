import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    L = math.sqrt(6/fan_in)

    output = [
        [
            W[i][j]*2*L - L for j in range(len(W[0]))
        ]
        for i in range(len(W))
    ]
    return output