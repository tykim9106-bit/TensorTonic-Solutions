import torch
import torch.nn.functional as F
import math

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    """
    Returns: dictionary containing output, hidden_width, and parameter_count
    """
    d = x.shape[-1]
    h_max = w_g.shape[-1]
    
    h = math.floor(base_params / (3 * d) + 0.5)
    h = max(1, min(h, h_max))

    wg = w_g[:, :h] 
    wv = w_v[:, :h] 
    wo = w_o[:h, :]

    gate = F.silu(x @ wg)
    value = x @ wv
    hidden = gate * value

    output = hidden @ wo

    return { "output": output, "hidden_width": int(h), "parameter_count": int(3 * d * h), }
    