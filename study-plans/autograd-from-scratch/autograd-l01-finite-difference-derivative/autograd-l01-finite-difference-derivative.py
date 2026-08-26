import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coefficients = coefficients[::-1]
    coefficients = np.asarray(coefficients, dtype = float)

    px = np.polyval(coefficients, x)
    px_h = np.polyval(coefficients, x+h)

    slope = (px_h - px)/h
    return px, px_h, slope
