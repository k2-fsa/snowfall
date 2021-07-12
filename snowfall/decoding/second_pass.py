# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

# This file implements the ideas proposed by Daniel Povey.
#
# See https://github.com/k2-fsa/snowfall/issues/232 for more details
#

import k2


class Nbest(object):
    '''
    An Nbest object contains two fields:

        (1) fsa, its type is k2.Fsa
        (2) shape, its type k2.RaggedShape (alias to _k2.RaggedShape)

    The field `fsa` is an FsaVec, containing a vector of linear FSAs.

    The field `shape` has two axes [seq][path]. `shape.dim0()` contains
    the number of sequences, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.
    '''

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        assert len(fsa.shape) == 3
        assert shape.num_axes() == 2
        assert fsa.shape[0] == shape.tot_size(1)

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = 'Nbest('
        s += f'num_seqs:{self.shape.dim0()}, '
        s += f'num_fsas:{self.fsa.shape[0]})'
        return s


def whole_lattice_rescoring(lats: k2.Fsa,
                            G_with_epsilon_loops: k2.Fsa) -> k2.Fsa:
    '''Rescore the 1st pass lattice with an LM.

    In general, the G in HLG used to obtain `lats` is a 3-gram LM.
    This function replaces the 3-gram LM in `lats` with a 4-gram LM.

    Args:
      lats:
        The decoding lattice from the 1st pass. We assume it is the result
        of intersecting HLG with the network output.
      G_with_epsilon_loops:
        An LM. It is usually a 4-gram LM with epsilon self-loops.
    Returns:
      Return a new lattice rescored with a given G.
    '''
    assert len(lats.shape) == 3
    assert hasattr(lats, 'lm_scores')
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lats.device
    lats.scores = lats.scores - lats.lm_scores
    # Now lats contains only acoustic scores

    # We will use lm_scores from G, so remove lats.lm_scores here
    del lats.lm_scores
    assert hasattr(lats, 'lm_scores') is False

    # inverted_lats has word IDs as labels.
    # Its aux_labels are phone IDs, which is a ragged tensor k2.RaggedInt
    inverted_lats = k2.invert(lats)
    num_seqs = lats.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)

    while True:
        try:
            rescoring_lats = k2.intersect_device(G_with_epsilon_loops,
                                                 inverted_lats,
                                                 b_to_a_map,
                                                 sorted_match_a=True)
            break
        except RuntimeError as e:
            logging.info(f'Caught exception:\n{e}\n')
            # Usually, this is an OOM exception. We reduce
            # the size of the lattice and redo k2.intersect_device()

            # NOTE(fangjun): The choice of the threshold 1e-5 is arbitrary here
            # to avoid OOM. We may need to fine tune it.
            inverted_lats = k2.prune_on_arc_post(inverted_lats, 1e-5, True)

    rescoring_lats = k2.top_sort(k2.connect(rescoring_lats))

    # inv_rescoring_lats has phone IDs as labels
    # and word IDs as aux_labels.
    inv_rescoring_lats = k2.invert(rescoring_lats)
    return inv_rescoring_lats


def generate_nbest_list(lats: k2.Fsa, num_paths: int) -> Nbest:
    '''Generate a nbest list from a lattice.

    Args:
      lats:
        The decoding lattice from the first pass after LM rescoring.
        lats is an FsaVec.
      num_paths:
        Size of n for n-best list. CAUTION: After removing paths
        that represent the same token sequences, not all sequences
        contain `n` paths.
    '''
    pass
