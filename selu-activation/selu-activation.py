import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    lambda_ = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717

    return [
        round(
            lambda_ * (
                value if value > 0
                else alpha * (math.exp(value) - 1)
            ),
            4
        )
        for value in x
    ]