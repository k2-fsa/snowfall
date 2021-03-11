#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)
#

from typing import Tuple

import torch

import k2


def create_phone_fsas(phone_seqs: k2.RaggedInt) -> k2.Fsa:
    '''
    Args:
      phone_seqs:
        It contains two axes with elements being phone IDs.
        The last element of each sub-list is -1. It does not
        contain 0s.
    Returns:
      Return an FsaVec representing the phone seqs.
    '''
    assert phone_seqs.num_axes() == 2
    phone_seqs = k2.ragged.remove_values_eq(phone_seqs, -1)
    return k2.linear_fsa(phone_seqs)


def generate_nbest_list_phone_seqs(lats: k2.Fsa,
                                   num_paths: int = 3,
                                   use_double_scores: bool = True
                                  ) -> k2.RaggedInt:  # noqa
    '''Generate n-best lists as a 3-D tensor containing phones with indexes
    [seq][nbest][phone_id].

    Args:
      lats:
        It has indexes [seq][states][arcs]
      num_paths:
        Size of `n` in the n-best list.
      use_double_scores:
        True to use double precision in `k2.random_paths`;
        False to use single precision in `k2.random_paths`.
    Returns:
      A 3-D tensor with indexes [seq][nbest][phone_id]. The last
      entry of every sublist is -1. The returned tensor does not
      contain 0s.
    '''
    assert len(lats.shape) == 3
    assert hasattr(lats, 'phones')

    # paths will be k2.RaggedInt with 3 axes: [seq][path][arc_pos],
    # containing arc_idx012
    paths = k2.random_paths(lats,
                            use_double_scores=use_double_scores,
                            num_paths=num_paths)

    # phone_seqs will be k2.RaggedInt like paths, but containing phones
    # (and final -1's, and 0's for epsilon)
    phone_seqs = k2.index(lats.phones, paths)

    # Remove epsilons from `phone_seqs`
    phone_seqs = k2.ragged.remove_values_eq(phone_seqs, 0)

    return phone_seqs


@torch.no_grad()
def compute_expected_times(
        lats: k2.Fsa,
        phone_seqs: k2.RaggedInt,
        ctc_topo: k2.Fsa,
        dense_fsa_vec: k2.DenseFsaVec,
        use_double_scores: bool = True,
        debug: bool = False
) -> Tuple[torch.Tensor, k2.Fsa, torch.Tensor, k2.RaggedInt]:
    '''
    Args:
      lats:
        It has indexes [seq][states][arcs]
      phone_seqs:
        It has indexes [seq][nbest][phone_id].
        See also :func:`generate_nbest_list_phone_seqs`.
        Or it is from the second pass decoding, which has only two axes.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      dense_fsa_vec:
        It contains nnet_output.
      use_double_scores:
        True to use double precision in `get_arc_post`;
        False to use single precision.
      debug:
        Some checks are enabled if `debug` is True.
    Returns:
      Return a tuple containing 4 entries:
        - expected_times, 1-D torch.Tensor indexed by pathphone_idx
        - phone_fsas, an FsaVec with indexes [path][phones]
        - path_to_seq_map, 1-D torch.Tensor
        - num_repeats, a k2.RaggedInt with 2 axes [path][multiplicities]
    '''
    device = lats.device

    if phone_seqs.num_axes() == 3:
        phone_seqs, num_repeats = k2.ragged.unique_sequences(phone_seqs, True)

        # Remove the 1st axis from `phone_seqs` (that corresponds to `seq`) and
        # keep it for later; we'll be processing paths separately.
        seq_to_path_shape = k2.ragged.get_layer(phone_seqs.shape(), 0)
        path_to_seq_map = seq_to_path_shape.row_ids(1)

        phone_seqs = k2.ragged.remove_axis(phone_seqs, 0)
    else:
        # we assume that phone_seqs is a 1-best list from the second pass decoding
        assert phone_seqs.num_axes() == 2
        num_repeats_value = torch.ones(phone_seqs.dim0(),
                                       dtype=torch.int32,
                                       device=device)
        num_repeats_shape = k2.ragged.regular_ragged_shape(
            phone_seqs.dim0(), 1).to(device)
        num_repeats = k2.RaggedInt(num_repeats_shape, num_repeats_value)

        seq_to_path_shape = num_repeats_shape
        path_to_seq_map = seq_to_path_shape.row_ids(1)  # an identity map

    # now compile decoding graphs corresponding to `phone_seqs` by constructing
    # fsas from them (remember we already have the final -1's!) and composing
    # with ctc_topo.
    phone_fsas = create_phone_fsas(phone_seqs)
    phone_fsas = k2.add_epsilon_self_loops(phone_fsas)

    # Set an attribute called pathphone_idx, which corresponds to the arc-index
    # in `phone_fsas` with self-loops.
    # Each phone has an index but there are blanks between them and at the
    # start and end.
    phone_fsas.pathphone_idx = torch.arange(phone_fsas.arcs.num_elements(),
                                            dtype=torch.int32,
                                            device=device)

    # Now extract the sets of paths from the lattices corresponding to each of
    # those n-best phone sequences; these will effectively be lattices with one
    # path but alternative alignments.
    path_decoding_graphs = k2.compose(ctc_topo,
                                      phone_fsas,
                                      treat_epsilons_specially=False)

    # CAUTION(fangjun): the supervision in dense_fsa_vec
    # has to be sorted in descending order!
    paths_lats = k2.intersect_dense(path_decoding_graphs,
                                    dense_fsa_vec,
                                    output_beam=10.0,
                                    a_to_b_map=path_to_seq_map,
                                    seqframe_idx_name='seqframe_idx')

    # by seq we mean the original sequence indexes, by path we mean the indexes
    # of the n-best paths; path_to_seq_map maps from path-index to seq-index.
    seqs_shape = dense_fsa_vec.dense_fsa_vec.shape()

    # paths_shape will also be a k2.RaggedInt with 2 axes
    paths_shape, _ = k2.ragged.index(seqs_shape,
                                     path_to_seq_map,
                                     need_value_indexes=False)

    seq_starts = seqs_shape.row_splits(1)[:-1]
    path_starts = paths_shape.row_splits(1)[:-1]

    # We can map from seqframe_idx  for paths, to seqframe_idx for seqs,
    # by adding path_offsets.  path_offsets is indexed by path-index.
    path_offsets = path_starts - k2.index(seq_starts, path_to_seq_map)

    paths_lats_arc2path = k2.index(paths_lats.arcs.shape().row_ids(1),
                                   paths_lats.arcs.shape().row_ids(2))

    # assign new attribute 'pathframe_idx' that combines path and frame.
    paths_lats.pathframe_idx = paths_lats.seqframe_idx + k2.index(
        path_offsets, paths_lats_arc2path)

    pathframe_to_pathphone = k2.create_sparse(
        rows=paths_lats.pathframe_idx,
        cols=paths_lats.pathphone_idx,
        values=paths_lats.get_arc_post(use_double_scores=use_double_scores,
                                       log_semiring=True).exp(),
        min_col_index=0)
    if debug:
        sum_per_row = torch.sparse.sum(pathframe_to_pathphone,
                                       dim=1).to_dense()
        expected_sum_per_row = torch.ones_like(sum_per_row)
        assert torch.allclose(sum_per_row, expected_sum_per_row)

    frame_idx = torch.arange(paths_shape.num_elements(),
                             device=device) - k2.index(path_starts,
                                                       paths_shape.row_ids(1))

    # TODO(fangjun): we can swap `rows` and `cols`
    # while creating `pathframe_to_pathphone` so that
    # `t()` can be omitted here.
    weighted_occupation = torch.sparse.mm(
        pathframe_to_pathphone.t(),
        frame_idx.unsqueeze(-1).to(pathframe_to_pathphone.dtype))
    # weighted_occupation has shape (num_pathphones, 1)

    # sum over columns to get the total occupation
    # Use `to_dense()` here because PyTorch does not
    # support div(dense_matrix, sparse_matrix)
    total_occupation = torch.sparse.sum(pathframe_to_pathphone,
                                        dim=0).to_dense()
    # total_occupation has shape (num_pathphone_idx, )

    eps = torch.finfo(total_occupation.dtype).eps
    expected_times = weighted_occupation.squeeze() / (total_occupation + eps)
    # expected_times has shape (num_pathphone_idx,)

    # Number of `pathphone_idx`'s should be even
    assert expected_times.shape[0] & 1 == 0

    # Replace the expected_times for even `pathphone_idx`'s with
    # the average expected times of two neighboring phones
    #
    # Even `pathphone_idx`'s belong to epsilon self-loops.
    expected_times[2::2] = (expected_times[1:-1:2] +
                            expected_times[3::2]) * 0.5
    # CAUTION: the above assignment is incorrect for the first epsilon
    # self-loop of the phone_fsas. The following statement assigns 0
    # to the first epsilon self-loop of every phone_fsa
    first_epsilon_offset = k2.index(phone_fsas.arcs.row_splits(2),
                                    phone_fsas.arcs.row_splits(1))

    # TODO(fangjun): do we need to support `torch.int32` for the indexing
    expected_times[first_epsilon_offset[:-1].long()] = 0

    return expected_times, phone_fsas, path_to_seq_map, num_repeats


def compute_embeddings_from_nnet_output(expected_times: torch.Tensor,
                                        phone_fsas: k2.Fsa,
                                        dense_fsa_vec: k2.DenseFsaVec,
                                        path_to_seq_map: torch.Tensor
                                       ) -> torch.Tensor:
    '''
    Args:
      expected_times:
        1-D torch.Tensor.
      phone_fsas:
        An FsaVec returned by :func:`compute_expected_times`.
      dense_fsa_vec:
        It contains nnet_output.
      path_to_seq_map:
        An 1-D torch.Tensor returned by :func:`compute_expected_times`.
    Returns:
      A 2-D torch.Tensor.
    '''
    # by seq we mean the original sequence indexes, by path we mean the indexes
    # of the n-best paths; path_to_seq_map maps from path-index to seq-index.
    seqs_shape = dense_fsa_vec.dense_fsa_vec.shape()
    seq_starts = seqs_shape.row_splits(1)[:-1]

    # paths_shape will also be a k2.RaggedInt with 2 axes
    paths_shape, _ = k2.ragged.index(seqs_shape,
                                     path_to_seq_map,
                                     need_value_indexes=False)

    pathphone_idx_to_path = k2.index(phone_fsas.arcs.row_ids(1),
                                     phone_fsas.arcs.row_ids(2))

    pathphone_idx_to_seq = k2.index(path_to_seq_map, pathphone_idx_to_path)

    first_epsilon_offset = k2.index(phone_fsas.arcs.row_splits(2),
                                    phone_fsas.arcs.row_splits(1))

    # linear interpolation
    #
    #                            y
    #    |-----------------------|-------------------|
    #   low   high_scale             low_scale      high
    #
    #  y = low * low_scale  + high * high_scale

    #  frame_idx_low = torch.floor(expected_times)
    # TODO(fangjun): change it to torch.floor
    # if interpolation is used
    frame_idx_low = torch.round(expected_times)

    #  frame_idx_high = torch.ceil(expected_times)

    #  low_scale = frame_idx_high - expected_times
    #  high_scale = 1 - low_scale
    offset = k2.index(seq_starts, pathphone_idx_to_seq)

    low = frame_idx_low + offset
    #  high = frame_idx_high + offset

    # the 0th column is not from the nnet_output and contains lots of -inf
    # so it is removed here
    scores = dense_fsa_vec.scores[:, 1:]
    low_scores = k2.index(scores, low.to(torch.int32))
    #  high_scores = k2.index(scores, high.to(torch.int32))

    # It can happen that `high_scores` contain fake nnet_output,
    # i.e., scores for arcs entering the final state are
    # [0, -inf, -inf, ..., ]
    #  embedding_scores = low_scores * low_scale.unsqueeze(
    #      -1) + high_scores * high_scale.unsqueeze(-1)

    # FIXME(fangjun): Use linear interpolation
    embedding_scores = low_scores

    # set `embedding_scores` of arcs entering final states to 0
    embedding_scores[(first_epsilon_offset[1:] - 1).long()] = 0

    return embedding_scores


def compute_embeddings_from_phone_fsas(phone_fsas: k2.Fsa,
                                       max_phone_id: int) -> torch.Tensor:
    '''Return one hot encodings of phones as embeddings.
    Args:
      phone_fsas:
        An FsaVec returned by :func:`compute_expected_times`.
      max_phone_id:
        The maximum phone ID.
    Return:
      A 2-D torch.Tensor.
    '''
    # phone_fsas is an acceptor, so its `labels` are `phones`
    pathphones = phone_fsas.labels.clone()

    device = phone_fsas.device
    # arcs entering the final state have phone == -1.
    # Increment it so that 0 represents EOS
    pathphones += 1
    num_classes = max_phone_id + 2  # +1 for the epsilon, +1 for EOS

    # TODO(fangjun): do we need to build our own one_hot that supports
    # dtype == torch.int32
    embedding_phones = torch.nn.functional.one_hot(
        pathphones.long(), num_classes=num_classes).to(device)

    return embedding_phones


def compute_embeddings_from_phone_seqs(
        lats: k2.Fsa,
        phone_seqs: k2.Fsa,
        ctc_topo: k2.Fsa,
        dense_fsa_vec: k2.DenseFsaVec,
        max_phone_id: int,
        use_double_scores: bool = True,
        debug: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, k2.RaggedInt]:
    '''
    Args:
      lats:
        An FsaVec.
      phone_seqs:
        It has indexes [seq][nbest][phone_id].
        See also :func:`generate_nbest_list_phone_seqs`.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      dense_fsa_vec:
        It contains nnet_output.
      max_phone_id:
        The maximum phone ID.
      use_double_scores:
        True to use double precision in :func:`k2.random_paths` and
        `get_arc_post`; False to use single precision.
      debug:
        Some checks are enabled if it is True.
    Returns:
      Return a tuple with 4 entries:
        - padded_embeddings, a 3-D torch.Tensor with dtype torch.float32
        - len_per_path, a 1-D CPU torch.Tensor
        - path_to_seq_map, a 1-D torch.Tensor
        - num_repeats, a ragged tensor of type k2.RaggedInt with 2
          axes [path][multiplicities]
    '''
    expected_times, phone_fsas, path_to_seq_map, num_repeats = compute_expected_times(  # noqa
        lats=lats,
        phone_seqs=phone_seqs,
        ctc_topo=ctc_topo,
        dense_fsa_vec=dense_fsa_vec,
        use_double_scores=use_double_scores,
        debug=debug)
    assert expected_times.requires_grad is False

    embedding_scores = compute_embeddings_from_nnet_output(
        expected_times, phone_fsas, dense_fsa_vec, path_to_seq_map)

    embedding_phones = compute_embeddings_from_phone_fsas(
        phone_fsas, max_phone_id)

    embeddings = torch.cat(
        (embedding_scores, embedding_phones, expected_times.unsqueeze(-1)),
        dim=1)

    first_epsilon_offset = k2.index(phone_fsas.arcs.row_splits(2),
                                    phone_fsas.arcs.row_splits(1))

    embeddings_per_path = []
    num_paths = phone_fsas.arcs.dim0()
    for i in range(num_paths - 1):
        start = first_epsilon_offset[i]
        end = first_epsilon_offset[i + 1]
        embeddings_per_path.append(embeddings[start:end])

    start = first_epsilon_offset[num_paths - 1]
    embeddings_per_path.append(embeddings[start:])
    len_per_path = first_epsilon_offset[1:] - first_epsilon_offset[0:-1]

    padded_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings_per_path,
                                                        batch_first=True)

    return padded_embeddings.to(
        torch.float32), len_per_path.cpu(), path_to_seq_map, num_repeats


def compute_embeddings(
        lats: k2.Fsa,
        ctc_topo: k2.Fsa,
        dense_fsa_vec: k2.DenseFsaVec,
        max_phone_id: int,
        use_double_scores: bool = True,
        num_paths: int = 3,
        debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, k2.RaggedInt]:
    '''
    Args:
      lats:
        An FsaVec.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      dense_fsa_vec:
        It contains nnet_output.
      max_phone_id:
        The maximum phone ID.
      use_double_scores:
        True to use double precision in :func:`k2.random_paths` and
        `get_arc_post`; False to use single precision.
      num_paths:
        Size of `n` in the n-best list.
      debug:
        Some checks are enabled if `debug` is True.
    Returns:
      Return a tuple with 4 entries:
        - padded_embeddings, a 3-D torch.Tensor with dtype torch.float32
        - len_per_path, a 1-D CPU torch.Tensor
        - path_to_seq_map, a 1-D torch.Tensor
        - num_repeats, a ragged tensor of type k2.RaggedInt with 2
          axes [path][multiplicities]
    '''
    phone_seqs = generate_nbest_list_phone_seqs(
        lats=lats, num_paths=num_paths, use_double_scores=use_double_scores)
    return compute_embeddings_from_phone_seqs(
        lats=lats,
        phone_seqs=phone_seqs,
        ctc_topo=ctc_topo,
        dense_fsa_vec=dense_fsa_vec,
        max_phone_id=max_phone_id,
        use_double_scores=use_double_scores,
        debug=debug)


def compute_embeddings_deprecated(
        lats: k2.Fsa,
        ctc_topo: k2.Fsa,
        dense_fsa_vec: k2.DenseFsaVec,
        max_phone_id: int,
        use_double_scores=True,
        num_paths=100,
        debug=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, k2.RaggedInt]:
    '''Compute embeddings for an n-best list.

    See the following comments for more information:

        - `<https://github.com/k2-fsa/snowfall/issues/96>`_
        - `<https://github.com/k2-fsa/k2/issues/641>`_

    Args:
      lats:
        An FsaVec.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      dense_fsa_vec:
        It contains nnet_output.
      max_phone_id:
        The maximum phone ID. Used for one-hot encoding.
      use_double_scores:
        True to use `double` in :func:`k2.random_paths`; false to use `float`.
      num_paths:
        Number of random paths to draw in :func:`k2.random_paths`.
      debug:
        Some checks are enabled if `debug` is True.
    Returns:
      Return a tuple with four tensors:
        - padded_embeddings, its shape is
          `(num_paths, max_phone_seq_len, num_features)`
        - len_per_path, its shape is (num_paths,) containing the phone_seq_len
          before padding
        - path_to_seq_map, its shape is (num_paths,)
        - num_repeats (k2.RaggedInt)
    '''
    device = lats.device
    assert len(lats.shape) == 3
    assert hasattr(lats, 'phones')

    # paths will be k2.RaggedInt with 3 axes: [seq][path][arc_pos],
    # containing arc_idx012
    paths = k2.random_paths(lats,
                            use_double_scores=use_double_scores,
                            num_paths=num_paths)
    if debug:
        assert paths.num_axes() == 3
        assert paths.num_elements() > 0

    #  print('paths', paths)

    # phone_seqs will be k2.RaggedInt like paths, but containing phones
    # (and final -1's, and 0's for epsilon)
    phone_seqs = k2.index(lats.phones, paths)
    if debug:
        assert phone_seqs.num_axes() == 3
        assert phone_seqs.num_elements() > 0

    #  print('lats.phones', lats.phones[:1000])
    #  print(phone_seqs)

    # Remove epsilons from `phone_seqs`
    #  print('before removing 0', phone_seqs.shape().row_splits(2))
    phone_seqs = k2.ragged.remove_values_eq(phone_seqs, 0)
    #  print('after removing 0', phone_seqs.shape().row_splits(2))

    # Remove repeated sequences from `phone_seqs`
    #
    phone_seqs, num_repeats = k2.ragged.unique_sequences(phone_seqs, True)

    # Remove the 1st axis from `phone_seqs` (that corresponds to `seq`) and
    # keep it for later, we'll be processing paths separately.
    seq_to_path_shape = k2.ragged.get_layer(phone_seqs.shape(), 0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    phone_seqs = k2.ragged.remove_axis(phone_seqs, 0)
    if debug:
        assert phone_seqs.num_axes() == 2

    # now compile decoding graphs corresponding to `phone_seqs` by constructing
    # fsas from them (remember we already have the final -1's!) and composing
    # with ctc_topo.
    phone_fsas = create_phone_fsas(phone_seqs)
    phone_fsas = k2.add_epsilon_self_loops(phone_fsas)
    if debug:
        assert phone_fsas.arcs.num_axes() == 3
        assert phone_fsas.arcs.num_elements() & 1 == 0

    # Set an attribute called pathphone_idx, which corresponds to the arc-index
    # in `phone_fsas` with self-loops.
    # Each phone has an index but there are blanks between them and at the
    # start and end.
    phone_fsas.pathphone_idx = torch.arange(phone_fsas.arcs.num_elements(),
                                            dtype=torch.int32,
                                            device=device)
    # phone_fsas is an acceptor, so its `labels` are `phones`
    pathphones = phone_fsas.labels.clone()

    pathphone_idx_to_path = k2.index(phone_fsas.arcs.row_ids(1),
                                     phone_fsas.arcs.row_ids(2))

    pathphone_idx_to_seq = k2.index(path_to_seq_map, pathphone_idx_to_path)

    # Now extract the sets of paths from the lattices corresponding to each of
    # those n-best phone sequences; these will effectively be lattices with one
    # path but alternative alignments.
    path_decoding_graphs = k2.compose(ctc_topo,
                                      phone_fsas,
                                      treat_epsilons_specially=False)

    paths_lats = k2.intersect_dense(path_decoding_graphs,
                                    dense_fsa_vec,
                                    output_beam=10.0,
                                    a_to_b_map=path_to_seq_map,
                                    seqframe_idx_name='seqframe_idx')

    # by seq we mean the original sequence indexes, by path we mean the indexes
    # of the n-best paths; path_to_seq_map maps from path-index to seq-index.
    seqs_shape = dense_fsa_vec.dense_fsa_vec.shape()

    # paths_shape will also be a k2.RaggedInt with 2 axes
    paths_shape, _ = k2.ragged.index(seqs_shape,
                                     path_to_seq_map,
                                     need_value_indexes=False)

    seq_starts = seqs_shape.row_splits(1)[:-1]
    path_starts = paths_shape.row_splits(1)[:-1]

    # We can map from seqframe_idx  for paths, to seqframe_idx for seqs,
    # by adding path_offsets.  path_offsets is indexed by path-index.
    path_offsets = path_starts - k2.index(seq_starts, path_to_seq_map)

    # assign new attribute 'pathframe_idx' that combines path and frame.
    paths_lats_arc2path = k2.index(paths_lats.arcs.shape().row_ids(1),
                                   paths_lats.arcs.shape().row_ids(2))

    paths_lats.pathframe_idx = paths_lats.seqframe_idx + k2.index(
        path_offsets, paths_lats_arc2path)

    pathframe_to_pathphone = k2.create_sparse(rows=paths_lats.pathframe_idx,
                                              cols=paths_lats.pathphone_idx,
                                              values=paths_lats.get_arc_post(
                                                  True, True).exp(),
                                              min_col_index=0)

    if debug:
        sum_per_row = torch.sparse.sum(pathframe_to_pathphone,
                                       dim=1).to_dense()
        expected_sum_per_row = torch.ones_like(sum_per_row)
        assert torch.allclose(sum_per_row, expected_sum_per_row)

    frame_idx = torch.arange(paths_shape.num_elements(),
                             device=device) - k2.index(path_starts,
                                                       paths_shape.row_ids(1))

    # TODO(fangjun): we can swap `rows` and `cols`
    # while creating `pathframe_to_pathphone` so that
    # `t()` can be omitted here.
    weighted_occupation = torch.sparse.mm(
        pathframe_to_pathphone.t(),
        frame_idx.unsqueeze(-1).to(pathframe_to_pathphone.dtype))
    # weighted_occupation has shape (num_pathphones, 1)

    # sum over columns to get the total occupation
    # Use `to_dense()` here because PyTorch does not
    # support div(dense_matrix, sparse_matrix)
    total_occupation = torch.sparse.sum(pathframe_to_pathphone,
                                        dim=0).to_dense()
    # total_occupation has shape (num_pathphone_idx, )

    eps = torch.finfo(total_occupation.dtype).eps
    expected_times = weighted_occupation.squeeze() / (total_occupation + eps)
    # expected_times has shape (num_pathphone_idx,)

    # Number of `pathphone_idx`'s should be even
    assert expected_times.shape[0] & 1 == 0

    # Replace the expected_times for even `pathphone_idx`'s with
    # the average expected times of two neighboring phones
    #
    # Even `pathphone_idx`'s belong to epsilon self-loops.
    expected_times[2::2] = (expected_times[1:-1:2] +
                            expected_times[3::2]) * 0.5
    # CAUTION: the above assignment is incorrect for the first epsilon
    # self-loop of the phone_fsas. The following statement assigns 0
    # to the first epsilon self-loop of every phone_fsa
    first_epsilon_offset = k2.index(phone_fsas.arcs.row_splits(2),
                                    phone_fsas.arcs.row_splits(1))

    # TODO(fangjun): do we need to support `torch.int32` for the indexing
    expected_times[first_epsilon_offset[:-1].long()] = 0

    if debug:
        # expected_times within a phone_fsa should be monotonic increasing
        assert expected_times.shape[0] == pathphone_idx_to_path.shape[0]

        for n in range(1, expected_times.shape[0]):
            if pathphone_idx_to_path[n] == pathphone_idx_to_path[n - 1]:
                assert expected_times[n] >= expected_times[
                    n - 1], f'{expected_times[n]}, {expected_times[n-1]}, {n}'
            #  else:
            #      # n is the first pathphone_idx of this fsa
            #      seq = pathphone_idx_to_seq[n - 1]
            #      seq_len = seqs_shape.row_splits(1)[
            #          seq + 1] - seqs_shape.row_splits(1)[seq]
            #
            #      assert pathphone_idx_to_path[
            #          n -
            #   1] < seq_len, f'{pathphone_idx_to_path[n - 1]}, {seq_len}, {n}'

        #  n = expected_times.shape[0] - 1
        #  seq = pathphone_idx_to_seq[n]
        #  seq_len = seqs_shape.row_splits(1)[seq +
        #                                    1] - seqs_shape.row_splits(1)[seq]
        #
        #  assert pathphone_idx_to_path[n] < seq_len

    # TODO(fangjun): we can remove the columns of even pathphone_idx
    # while constructing `pathframe_to_pathphone`, which can save about
    # half computation time in `torch.sparse.mm`.

    # linear interpolation
    #
    #                            y
    #    |-----------------------|-------------------|
    #   low   high_scale             low_scale      high
    #
    #  y = low * low_scale  + high * high_scale

    #  frame_idx_low = torch.floor(expected_times)
    # TODO(fangjun): change it to torch.floor
    # if interpolation is used
    frame_idx_low = torch.round(expected_times)
    #  frame_idx_high = torch.ceil(expected_times)

    #  low_scale = frame_idx_high - expected_times
    #  high_scale = 1 - low_scale
    offset = k2.index(seq_starts, pathphone_idx_to_seq)

    low = frame_idx_low + offset
    #  high = frame_idx_high + offset

    # the first column is not from the nnet_output and contains lots of -inf
    # so it is removed here
    scores = dense_fsa_vec.scores[:, 1:]
    low_scores = k2.index(scores, low.to(torch.int32))
    #  high_scores = k2.index(scores, high.to(torch.int32))

    # It can happen that `high_scores` contain fake nnet_output,
    # i.e., scores for arcs entering the final state are
    # [0, -inf, -inf, ..., ]
    #  embedding_scores = low_scores * low_scale.unsqueeze(
    #      -1) + high_scores * high_scale.unsqueeze(-1)

    # FIXME(fangjun): Use linear interpolation
    embedding_scores = low_scores

    # set `embedding_scores` of arcs entering final states to 0
    embedding_scores[(first_epsilon_offset[1:] - 1).long()] = 0

    # arcs entering the final state have phone == -1.
    # Increment it so that 0 represents EOS
    pathphones += 1
    num_classes = max_phone_id + 2  # +1 for the epsilon, +1 for EOS
    #  print(pathphones.max(), pathphones.min(), num_classes)

    # TODO(fangjun): do we need to build our own one_hot
    # that supports dtype == torch.int32
    embedding_phones = torch.nn.functional.one_hot(
        pathphones.long(), num_classes=num_classes).to(device)

    #  print(embedding_scores.shape)
    #  print(embedding_phones.shape)
    #  print(expected_times.unsqueeze(-1).shape)
    embeddings = torch.cat(
        (embedding_scores, embedding_phones, expected_times.unsqueeze(-1)),
        dim=1)

    if debug:
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == pathphones.shape[0]
        assert embeddings.shape[1] == (dense_fsa_vec.scores.shape[1] - 1 +
                                       num_classes + 1)
        assert embeddings.shape[0] == phone_fsas.arcs.row_splits(2)[-1]

    embeddings_per_path = []
    num_paths = phone_fsas.arcs.dim0()
    for i in range(num_paths - 1):
        start = first_epsilon_offset[i]
        end = first_epsilon_offset[i + 1]
        embeddings_per_path.append(embeddings[start:end])

    start = first_epsilon_offset[num_paths - 1]
    embeddings_per_path.append(embeddings[start:])
    len_per_path = first_epsilon_offset[1:] - first_epsilon_offset[0:-1]

    padded_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings_per_path,
                                                        batch_first=True)

    if debug:
        s = 0
        for p in embeddings_per_path:
            s += p.shape[0]
        assert s == len_per_path.sum().item()
        assert s == embeddings.shape[0]

    # It used `double` for `get_arc_post`, but the network input requires
    # torch.float32
    return padded_embeddings.to(
        torch.float32), len_per_path.cpu(), path_to_seq_map, num_repeats
