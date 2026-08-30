def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns: list[int] containing token IDs after applying the ordered merge rules
    """
    ids = list(text.encode("utf-8"))

    for left_id, right_id, next_id in merges:
        new_ids =[]
        i = 0

        while i < len(ids):
            if(
                i < len(ids)-1 and
                ids[i] == left_id and
                ids[i+1] == right_id
            ):
                new_ids.append(next_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        ids = new_ids

    return ids

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns: the Unicode string reconstructed from token IDs and vocabulary bytes
    """
    byte_values = []

    for token_id in ids:
        byte_values.extend(vocab[token_id])

    return bytes(byte_values).decode("utf-8")