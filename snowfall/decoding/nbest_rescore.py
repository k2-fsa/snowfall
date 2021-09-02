# Copyright (c)  2021  Xiaomi Corporation (authors: Wei Kang)

import k2
import torch


def get_best_matching_stats(keys: k2.Nbest, queries: k2.Nbest,
                            max_order: int) -> torch.Tensor:
    '''Get best matching stats on query positions.

    Args:
      keys:
        The nbest after doing second pass rescoring.
      queries:
        Another nbest before doing second pass rescoring.
      max_order:
        The maximum n-gram order to ever return by `k2.get_best_matching_stats`

    Returns:
      A tensor with the shape of [queries.fsa.num_elements, 5], each row
      contains the stats (init_score, mean, var, counts_out, ngram_order)
      of the token in the correspodding position in queries.
    '''
    assert keys.shape.dim0() == queries.shape.dim0(), \
        f'Utterances number in keys and queries should be equal : \
         {keys.shape.dim0()} vs {queries.shape.dim0()}'

    # keys_tokens_shape [utt][path][token]
    keys_tokens_shape = k2.ragged.compose_ragged_shapes(keys.shape,
        k2.ragged.remove_axis(keys.fsa.arcs.shape(), 1))
    # queries_tokens_shape [utt][path][token]
    queries_tokens_shape = k2.ragged.compose_ragged_shapes(queries.shape,
        k2.ragged.remove_axis(queries.fsa.arcs.shape(), 1))

    keys_tokens = k2.RaggedInt(keys_tokens_shape, keys.fsa.labels.clone())
    queries_tokens = k2.RaggedInt(queries_tokens_shape,
                                  queries.fsa.labels.clone())
    # tokens shape [utt][path][token]
    tokens = k2.ragged.cat([keys_tokens, queries_tokens], axis=1)

    keys_token_num = keys.fsa.labels.size()[0]
    queries_tokens_num = queries.fsa.labels.size()[0]
    # counts on key positions are ones
    keys_counts = k2.RaggedInt(keys_tokens_shape,
                               torch.ones(keys_token_num,
                                          dtype=torch.int32))
    # counts on query positions are zeros
    queries_counts = k2.RaggedInt(queries_tokens_shape,
                                  torch.zeros(queries_tokens_num,
                                              dtype=torch.int32))
    counts = k2.ragged.cat([keys_counts, queries_counts], axis=1).values()

    # scores on key positions are the scores inherted from nbest path
    keys_scores = k2.RaggedFloat(keys_tokens_shape, keys.fsa.scores.clone())
    # scores on query positions MUST be zeros
    queries_scores = k2.RaggedFloat(queries_tokens_shape,
                                    torch.zeros(queries_tokens_num,
                                                dtype=torch.float32))
    scores = k2.ragged.cat([keys_scores, queries_scores], axis=1).values()

    # we didn't remove -1 labels before
    min_token = -1
    eos = -1
    max_token = torch.max(torch.max(keys.fsa.labels),
                          torch.max(queries.fsa.labels))
    mean, var, counts_out, ngram = k2.get_best_matching_stats(tokens, scores,
        counts, eos, min_token, max_token, max_order)

    queries_init_scores = queries.fsa.scores.clone()
    # only return the stats on query positions
    masking = counts == 0
    # shape [queries_tokens_num, 5]
    return torch.transpose(torch.stack((queries_init_scores, mean[masking],
                                        var[masking], counts_out[masking],
                                        ngram[masking])), 0, 1)


if __name__ == '__main__':
    fsa1 = k2.linear_fsa([4, 6, 7, 1])
    fsa1.scores = torch.tensor([2, 3, 4, 1, 0], dtype=torch.float32)
    fsa2 = k2.linear_fsa([4, 3, 7, 1])
    fsa2.scores = torch.tensor([2, 1.5, 4, 1, 0], dtype=torch.float32)
    fsa3 = k2.linear_fsa([4, 3, 2, 1])
    fsa3.scores = torch.tensor([2, 3, 2.5, 1, 0], dtype=torch.float32)
    fsa4 = k2.linear_fsa([5, 6, 7, 1])
    fsa4.scores = torch.tensor([0.5, 3, 4, 1, 0], dtype=torch.float32)
    fsa5 = k2.linear_fsa([12, 8, 11])
    fsa5.scores = torch.tensor([1.5, 3.5, 1, 0], dtype=torch.float32)
    fsa6 = k2.linear_fsa([12, 8, 9])
    fsa6.scores = torch.tensor([1.5, 3.5, 2, 0], dtype=torch.float32)
    fsa7 = k2.linear_fsa([12, 10, 11])
    fsa7.scores = torch.tensor([1.5, 3, 4, 0], dtype=torch.float32)
    # construct key nbest
    key_fsas = k2.create_fsa_vec([fsa1, fsa2, fsa5, fsa6])
    key_row_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    key_shape = k2.ragged.create_ragged_shape2(row_ids=key_row_ids)
    keys = k2.Nbest(key_fsas, key_shape)
    # construct queries nbest
    query_fsas = k2.create_fsa_vec([fsa3, fsa4, fsa7])
    query_row_ids = torch.tensor([0, 0, 1], dtype=torch.int32)
    query_shape = k2.ragged.create_ragged_shape2(row_ids=query_row_ids)
    queries = k2.Nbest(query_fsas, query_shape)
    # when calling get_best_matching_stats, scores in queries are initial
    # scores (before second pass rescoring)
    feature = get_best_matching_stats(keys, queries, 3)
    # we should do second pass rescoring on queries here.
    # the ground truth (last column) should be the scores after second pass
    # rescoring.
    training_data = torch.cat((feature,
                               queries.fsa.scores.unsqueeze(1)), dim=1)
    print(training_data)
    torch.save(training_data, "./nbest.pt")
