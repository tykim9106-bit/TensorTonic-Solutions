import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    PE = np.zeros((seq_length, d_model))

    pos = np.arange(seq_length)[:,None]

    div_term = np.exp(
        np.arange(0, d_model,2) * -np.log(10000.0)/d_model
    )
    PE[:,0::2] = np.sin(pos*div_term)
    PE[:,1::2] = np.cos(pos*div_term)

    return PE