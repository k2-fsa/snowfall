from pathlib import Path

import logging

import k2
import torch

from snowfall.common import find_first_disambig_symbol
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols


def compile_HLG(
        L: k2.Fsa,
        G: k2.Fsa,
        H: k2.Fsa,
        labels_disambig_id_start: int,
        aux_labels_disambig_id_start: int
) -> k2.Fsa:
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
        H:  An ``Fsa`` that represents a specific topology used to convert the network
            outputs to a sequence of phones.
            Typically, it's a CTC topology fst, in which when 0 appears on the left
            side, it represents the blank symbol; when it appears on the right side,
            it indicates an epsilon.
        labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            phonetic alphabet.
        aux_labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            words vocabulary.
    :return:
    """
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)
    # Attach a new attribute `lm_scores` so that we can recover
    # the `am_scores` later.
    # The scores on an arc consists of two parts:
    #  scores = am_scores + lm_scores
    # NOTE: we assume that both kinds of scores are in log-space.
    G.lm_scores = G.scores.clone()

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting L*G")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[LG.aux_labels.values() >= aux_labels_disambig_id_start] = 0
    logging.info("Removing epsilons")
    LG = k2.remove_epsilon(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing ctc_topo LG")
    HLG = k2.compose(H, LG, inner_labels='phones')

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(
        f'LG is arc sorted: {(HLG.properties & k2.fsa_properties.ARC_SORTED) != 0}'
    )

    return HLG


def load_or_compile_HLG(lang_dir: str) -> k2.Fsa:
    '''Build an HLG graph from a given directory.

    The following files are expected to be available in
    the given directory:

        - HLG.pt
        - If HLG.pt does not exist, then the following must
          be available:

            - words.txt
            - phones.txt
            - L_disambig.fst.txt
            - G.fst.txt

    Args:
      lang_dir:
        Path to the language directory.
    Returns:
      Return an HLG graph.
    '''
    lang_dir = Path(lang_dir)

    if (lang_dir / 'HLG.pt').exists():
        logging.info('Loading pre-compiled HLG')
        d = torch.load(lang_dir / 'HLG.pt')
        HLG = k2.Fsa.from_dict(d)
        return HLG

    word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)

    phone_ids_with_blank = [0] + phone_ids
    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

    logging.info('Loading L_disambig.fst.txt')
    with open(lang_dir / 'L_disambig.fst.txt') as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False)

    logging.info('Loading G.fst.txt')
    with open(lang_dir / 'G.fst.txt') as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)

    first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)

    first_word_disambig_id = find_first_disambig_symbol(word_symbol_table)

    HLG = compile_HLG(L=L,
                      G=G,
                      H=ctc_topo,
                      labels_disambig_id_start=first_phone_disambig_id,
                      aux_labels_disambig_id_start=first_word_disambig_id)

    torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')

    return HLG
