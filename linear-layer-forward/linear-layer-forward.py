def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    outputs = []
    for i in range(len(X)):
        output =[]
        for j in range(len(W[0])):
            value = 0
            for k in range(len(W)):
                value += X[i][k] * W[k][j]
            output.append(value + b[j])
        outputs.append(output)

    return outputs