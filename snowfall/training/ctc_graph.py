# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import (
    Iterable,
    List,
)

import torch
import k2
import _k2


def build_ctc_topo(phones: List[int]) -> k2.Fsa:
    '''Build CTC topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Args:
      phones:
        A list of phones
    Returns:
      Returns an FSA that converts repeated symbols to a single symbol.
    '''
    num_state = len(phones)
    final_state = num_state
    rules = ''
    for i in range(num_state):
        for j in range(num_state):
            if i == j:
                rules += f'{i} {i} {phones[i]} 0 0.0\n'
            else:
                rules += f'{i} {j} {phones[j]} {phones[j]} 0.0\n'
        rules += f'{i} {final_state} -1 -1 0.0\n'
    rules += f'{final_state}'
    ans = k2.Fsa.from_str(rules)
    return ans


def compose(a_fsa: k2.Fsa, b_fsa: k2.Fsa) -> k2.Fsa:
    '''Compose two graphs.

    Args:
      a_fsa:
        The FSA on the left hand side.
      b_fsa:
        The FSA on the right hand side.

    Returns:
      The composition result. It will have:
    '''
    if not hasattr(a_fsa, 'aux_labels'):
        return k2.intersect(a_fsa, b_fsa)

    if not hasattr(b_fsa, 'aux_labels'):
        return k2.intersect(a_fsa, b_fsa)

    a_fsa = a_fsa.invert()

    a_fsa = k2.arc_sort(a_fsa)
    b_fsa = k2.arc_sort(b_fsa)

    need_arc_map = True
    treat_epsilons_specially = True
    ragged_arc, a_arc_map, b_arc_map = _k2.intersect(
        a_fsa.arcs, a_fsa.properties, b_fsa.arcs, b_fsa.properties,
        treat_epsilons_specially, need_arc_map)

    out_fsa = k2.Fsa(ragged_arc)
    out_fsa.labels = k2.index_attr(a_fsa.aux_labels, a_arc_map)
    out_fsa.aux_labels = k2.index_attr(b_fsa.aux_labels, b_arc_map)

    for name, a_value in a_fsa.named_tensor_attr():
        if hasattr(b_fsa, name):
            # Both a_fsa and b_fsa have this attribute.
            # We only support attributes with dtype `torch.float32`.
            # Other kinds of attributes are discarded.
            if a_value.dtype != torch.float32:
                continue
            b_value = getattr(b_fsa, name)
            assert b_value.dtype == torch.float32

            value = k2.index_select(a_value, a_arc_map) + k2.index_select(
                b_value, b_arc_map)
            setattr(out_fsa, name, value)
        else:
            # only a_fsa has this attribute, copy it via arc_map
            value = k2.index_attr(a_value, a_arc_map)
            setattr(out_fsa, name, value)

    # now copy tensor attributes that are in b_fsa but are not in a_fsa
    for name, b_value in b_fsa.named_tensor_attr():
        if not hasattr(out_fsa, name):
            value = k2.index_attr(b_value, b_arc_map)
            setattr(out_fsa, name, value)

    for name, a_value in a_fsa.named_non_tensor_attr():
        if name == 'symbols':
            continue

        if name == 'aux_symbols':
            setattr(out_fsa, 'symbols', a_value)
        else:
            setattr(out_fsa, name, a_value)

    for name, b_value in b_fsa.named_non_tensor_attr():
        if not hasattr(out_fsa, name):
            setattr(out_fsa, name, b_value)

    return out_fsa


class CtcTrainingGraphCompiler:

    def __init__(self,
                 L_inv: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while the aux_labels are phones. Must
            be arc sorted.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        assert L_inv.properties & k2.fsa_properties.ARC_SORTED != 0

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov = oov
        self.ctc_topo = build_ctc_topo(list(phones._id2sym.keys()))

    def compile(self, texts: Iterable[str]) -> k2.Fsa:
        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        decoding_graph = compose(self.ctc_topo, decoding_graph)
        return decoding_graph
