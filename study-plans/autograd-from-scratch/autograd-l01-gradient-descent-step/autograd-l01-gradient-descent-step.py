import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype = float)
    gradients = np.asarray(gradients, dtype = float)

    updated_values = values - learning_rate * gradients
    delta_L = np.dot(gradients, (updated_values-values))

    return updated_values.tolist(), (delta_L)
