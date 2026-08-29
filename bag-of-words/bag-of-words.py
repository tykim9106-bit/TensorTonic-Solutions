import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    vocabindex = {
        word : index
        for index, word in enumerate(vocab)
    }
    output =np.zeros(len(vocab))

    for token in tokens:
        if token in vocabindex:
            output[vocabindex[token]] += 1

    return output