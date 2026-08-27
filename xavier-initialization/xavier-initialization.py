import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    L = math.sqrt(6/(fan_in+fan_out))

    output = [
        [W[i][j]*2*L - L for j in range(len(W[0]))]
        for i in range(len(W))
    ]
    return output