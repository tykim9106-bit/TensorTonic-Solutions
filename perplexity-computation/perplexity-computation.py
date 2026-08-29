import math

def perplexity(prob_distributions: list, actual_tokens: list) -> float:
    """
    Returns the sequence perplexity.
    """
    logsum = 0
    for i, actual_token in enumerate(actual_tokens):
        p_i = prob_distributions[i][actual_token]
        logsum += math.log(p_i)

    H = -logsum/len(actual_tokens)
    PP = math.exp(H)

    return PP
        