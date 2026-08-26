import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    def func_d (a,b,c):
        return a*b + c

    grad_a = (func_d(a+h, b, c)-func_d(a,b,c))/h
    grad_b = (func_d(a, b+h, c)-func_d(a,b,c))/h
    grad_c = (func_d(a, b, c+h)-func_d(a,b,c))/h

    return func_d(a, b, c), grad_a, grad_b, grad_c