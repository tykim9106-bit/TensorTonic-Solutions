def train_bpe(corpus, vocab_size):
    """
    Returns: dictionary containing learned vocabulary entries and ordered merges
    """
    sequences = [list(s.encode("utf-8")) for s in corpus]

    token_bytes = { i : [i] for i in range(256)}

    vocab = []
    merges =[]
    next_id = 256

    while next_id < vocab_size:
        pair_counts = {}

        for seq in sequences:
            for i in range(len(seq) -1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1


        if not pair_counts:
            break

        max_count = max(pair_counts.values())

        candidates =[
            pair for pair, count in pair_counts.items()
            if count == max_count
        ]

        best_pair = max(
            candidates,
            key = lambda pair: (
                token_bytes[pair[0]],
                token_bytes[pair[1]]
            )
        )

        left_id, right_id = best_pair

        new_bytes = token_bytes[left_id] + token_bytes[right_id]
        token_bytes[next_id] = new_bytes
        vocab.append([next_id, new_bytes.copy()])
        merges.append([left_id, right_id, next_id])

        for seq_idx, seq in enumerate(sequences):
            new_seq = []
            i = 0

            while i < len(seq) :
                if (
                    i<len(seq) - 1 and
                    seq[i] == left_id and
                    seq[i+1] == right_id
                    
                ):
                    new_seq.append(next_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            sequences[seq_idx] = new_seq
            
        next_id += 1

    return {
        "vocab" : vocab,
        "merges" : merges
    }