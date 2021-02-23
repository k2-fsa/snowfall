#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)
#

import k2
import torch


def _create_phone_fsas(phone_seqs: k2.RaggedInt) -> k2.Fsa:
    '''
    Args:
      phone_seqs:
        It contains two axes with elements being phone IDs.
        The last element of each sub-list is -1.
    Returns:
      Return an FsaVec representing the phone seqs.
    '''
    assert phone_seqs.num_axes() == 2
    phone_seqs = k2.ragged.remove_values_eq(phone_seqs, -1)
    return k2.linear_fsa(phone_seqs)


def compute_expected_times_per_phone(mbr_lats: k2.Fsa,
                                     ctc_topo: k2.Fsa,
                                     dense_fsa_vec: k2.DenseFsaVec,
                                     use_double_scores=True,
                                     num_paths=100) -> torch.Tensor:
    '''Compute expected times per phone in a n-best list.

    See the following comments for more information:

        - `<https://github.com/k2-fsa/snowfall/issues/96>`_
        - `<https://github.com/k2-fsa/k2/issues/641>`_

    Args:
      mbr_lats:
        An FsaVec.
      ctc_topo:
        The return value of :func:`build_ctc_topo`.
      dense_fsa_vec:
        It contains nnet_output.
      use_double_scores:
        True to use `double` in :func:`k2.random_paths`; false to use `float`.
      num_paths:
        Number of random paths to draw in :func:`k2.random_paths`.
    Returns:
      A 1-D torch.Tensor contains the expected times per pathphone_idx.
    '''
    lats = mbr_lats
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

    # Remove repeated sequences from `phone_seqs`
    #
    # TODO(fangjun): `num_repeats` is currently not used
    phone_seqs, num_repeats = k2.ragged.unique_sequences(phone_seqs, True)

    # Remove the 1st axis from `phone_seqs` (that corresponds to `seq`) and
    # keep it for later, we'll be processing paths separately.
    seq_to_path_shape = k2.ragged.get_layer(phone_seqs.shape(), 0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    phone_seqs = k2.ragged.remove_axis(phone_seqs, 0)

    # now compile decoding graphs corresponding to `phone_seqs` by constructing
    # fsas from them (remember we already have the final -1's!) and composing
    # with ctc_topo.
    phone_fsas = _create_phone_fsas(phone_seqs)
    phone_fsas = k2.add_epsilon_self_loops(phone_fsas)

    # Set an attribute called pathphone_idx, which corresponds to the arc-index
    # in `phone_fsas` with self-loops.
    # Each phone has an index but there are blanks between them and at the start
    # and end.
    phone_fsas.pathphone_idx = torch.arange(phone_fsas.arcs.num_elements(),
                                            dtype=torch.int32)

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

    frame_idx = torch.arange(paths_shape.num_elements()) - k2.index(
        path_starts, paths_shape.row_ids(1))

    # TODO(fangjun): we can swap `rows` and `cols`
    # while creating `pathframe_to_pathphone` so that
    # `t()` can be omitted here.
    weighted_occupation = torch.sparse.mm(
        pathframe_to_pathphone.t(),
        frame_idx.unsqueeze(-1).to(pathframe_to_pathphone.dtype))

    # sum over columns to get the total occupation
    # Use `to_dense()` here because PyTorch does not
    # support div(dense_matrix, sparse_matrix)
    total_occupation = torch.sparse.sum(pathframe_to_pathphone,
                                        dim=0).to_dense()

    return weighted_occupation.squeeze() / total_occupation
