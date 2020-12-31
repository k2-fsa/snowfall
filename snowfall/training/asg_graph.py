# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import Iterable
from typing import List

import re

import k2
import torch

from .ctc_graph import build_ctc_topo


def filter_disambig_symbols(symbol_table: k2.SymbolTable,
                            pattern: str = '#') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.

    Caution:
      You may need to remove the ID with value 0 if the return value
      is used to create a phone LM.

    Args:
      symbol_table:
        A symbol table in k2.
      pattern:
        Symbols containing this pattern are disambiguation symbols.
    Returns:
      Return a list of symbol IDs excluding those from disambiguation symbols.
    '''
    regex = re.compile(pattern)
    symbols = symbol_table.symbols
    ans = []
    for s in symbols:
        if not regex.search(s):
            ans.append(symbol_table[s])
    return ans


def create_bigram_phone_lm(phones: List[int]) -> k2.Fsa:
    '''Create a bigram phone LM.
    The resulting FSA (P) has a start-state and a state for
    each phone 0, 1, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.

    Caution:
      blank is not a phone.
    '''
    final_state = len(phones) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {phones[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {phones[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
    print(len(rules.split('\n')))
    return k2.Fsa.from_str(rules)


class AsgTrainingGraphCompiler(object):

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

        assert L_inv.requires_grad is False

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov = oov

        # note that `non_disambig_ids` contains 0, which represents
        # blank in the context of CTC.
        non_disambig_ids = filter_disambig_symbols(phones)

        ctc_topo = build_ctc_topo(non_disambig_ids)
        assert ctc_topo.requires_grad is False

        self.ctc_topo = ctc_topo

    def compile(self, texts: Iterable[str], P: k2.Fsa) -> k2.Fsa:
        assert P.requires_grad is True

        den = k2.intersect(self.ctc_topo, P).invert_()
        den = k2.connect(den)

        decoding_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])
        assert P.requires_grad is False

        num = k2.compose(den, decoding_graphs)
        num = k2.connect(num)
        num = k2.arc_sort(num)
        assert num.requires_grad is True

        return num, den.detach()

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        decoding_graph = k2.arc_sort(decoding_graph)
        return decoding_graph
