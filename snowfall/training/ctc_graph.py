# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import Iterable
from typing import List

import torch
import k2


def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    '''Build CTC topology.

    The resulting topology converts repeated input
    symbols to a single output symbol.

    Caution:
      The resulting topo is an FST. Epsilons are on the right
      side (i.e., olabels) and tokens are on the left side (i.e., ilabels)

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    '''
    assert 0 in tokens, 'We assume 0 is ID of the blank symbol'

    num_states = len(tokens)
    final_state = num_states
    arcs = ''
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f'{i} {i} {tokens[i]} 0 0.0\n'
            else:
                arcs += f'{i} {j} {tokens[j]} {tokens[j]} 0.0\n'
        arcs += f'{i} {final_state} -1 -1 0.0\n'
    arcs += f'{final_state}'
    ans = k2.Fsa.from_str(arcs)
    ans = k2.arc_sort(ans)
    return ans


class CtcTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''
        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov = oov
        self.ctc_topo = k2.arc_sort(build_ctc_topo(list(phones._id2sym.keys())))

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
        label_graph = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(label_graph,
                                                 self.L_inv)).invert_()
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = k2.compose(self.ctc_topo, decoding_graph)
        decoding_graph = k2.connect(decoding_graph)
        return decoding_graph
