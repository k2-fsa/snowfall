import logging

import k2
import torch
from k2 import Fsa


def compile_LG(L: Fsa, G: Fsa, labels_disambig_id_start: int,
               aux_labels_disambig_id_start: int) -> Fsa:
    """
    Creates a decoding graph using a lexicon fst ``L`` and language model fsa ``G``.
    Involves arc sorting, intersection, determinization, removal of disambiguation symbols
    and adding epsilon self-loops.

    Args:
        L:
            An ``Fsa`` that represents the lexicon (L), i.e. has phones as ``symbols``
                and words as ``aux_symbols``.
        G:
            An ``Fsa`` that represents the language model (G), i.e. it's an acceptor
            with words as ``symbols``.
        labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            phonetic alphabet.
        aux_labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            words vocabulary.
    :return:
    """
    L_inv = k2.arc_sort(L.invert_())
    G = k2.arc_sort(G)
    logging.debug("Intersecting L and G")
    LG = k2.intersect(L_inv, G)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Connecting L*G")
    LG = k2.connect(LG).invert_()
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[LG.aux_labels.values() >= aux_labels_disambig_id_start] = 0
    logging.debug("Removing epsilons")
    LG = k2.remove_epsilons_iterative_tropical(LG)
    logging.debug(f'LG shape = {LG.shape}')
    logging.debug("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.debug(f'LG shape = {LG.shape}')
    LG = k2.add_epsilon_self_loops(LG)
    LG = k2.arc_sort(LG)
    logging.debug(
        f'LG is arc sorted: {(LG.properties & k2.fsa_properties.ARC_SORTED) != 0}'
    )
    return LG
