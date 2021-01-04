# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import Iterable
from typing import List
from typing import Tuple

import re

import k2
import torch

from .ctc_graph import build_ctc_topo


def get_phone_symbols(symbol_table: k2.SymbolTable,
                      pattern: str = r'^#\d+$') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.

    Caution:
      0 is not a phone ID so it is excluded from the return value.

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
        if not regex.match(s):
            ans.append(symbol_table[s])
    if 0 in ans:
        ans.remove(0)
    ans.sort()
    return ans


def create_bigram_phone_lm(phones: List[int]) -> k2.Fsa:
    '''Create a bigram phone LM.
    The resulting FSA (P) has a start-state and a state for
    each phone 1, 2, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.

    Caution:
      blank is not a phone.

    Args:
      A list of phone IDs.

    Returns:
      An FSA representing the bigram phone LM.
    '''
    assert 0 not in phones
    final_state = len(phones) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {phones[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {phones[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
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

        phone_symbols = get_phone_symbols(phones)
        phone_symbols_with_blank = [0] + phone_symbols

        ctc_topo = build_ctc_topo(phone_symbols_with_blank)
        assert ctc_topo.requires_grad is False

        self.ctc_topo = ctc_topo

    def compile(self, texts: Iterable[str],
                P: k2.Fsa) -> Tuple[k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph`.
        '''
        assert P.is_cpu()

        den = k2.intersect(self.ctc_topo, P).invert_()
        den = k2.connect(den)

        num_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        num = k2.compose(den, num_graphs)
        num = k2.connect(num)
        num = k2.arc_sort(num)

        den = k2.create_fsa_vec([den.detach()] * len(texts))

        return num, den

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> k2.Fsa:
        '''Convert transcript to an Fsa with the help of lexicon
        and word symbol table.

        Args:
          text:
            The transcript containing words separated by spaces.

        Returns:
          Return an FST corresponding to the transcript. Its `labels` are
          phone IDs and `aux_labels` are word IDs.
        '''
        tokens = (token if token in self.words else self.oov
                  for token in text.split(' '))
        word_ids = [self.words[token] for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        num_graph = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        num_graph = k2.arc_sort(num_graph)
        return num_graph
