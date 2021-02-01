# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

from functools import lru_cache
from typing import Iterable
from typing import List
from typing import Tuple
from pathlib import Path

import logging

import k2
import torch

from .ctc_graph import build_ctc_topo
from snowfall.common import get_phone_symbols
from snowfall.decoding.graph import compile_LG


def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))


class MmiMbrTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 L_disambig: k2.Fsa,
                 G: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
          L_disambig:
            L with disambig symbols. Its labels are phones and aux_labels
            are words.
          G:
            The language model.
          phones:
            The phone symbol table.
          words:
            The word symbol table.
          oov:
            Out of vocabulary word.
        '''

        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        if G.properties & k2.fsa_properties.ARC_SORTED != 0:
            G = k2.arc_sort(G)

        assert L_inv.requires_grad is False
        assert G.requires_grad is False

        assert oov in words

        L = L_inv.invert()
        L = k2.arc_sort(L)

        self.L_inv = L_inv
        self.L = L
        self.phones = phones
        self.words = words
        self.oov = oov

        phone_symbols = get_phone_symbols(phones)
        phone_symbols_with_blank = [0] + phone_symbols

        ctc_topo = k2.arc_sort(build_ctc_topo(phone_symbols_with_blank))
        assert ctc_topo.requires_grad is False

        self.ctc_topo = ctc_topo
        self.ctc_topo_inv = k2.arc_sort(ctc_topo.invert())

        lang_dir = Path('data/lang_nosp')
        if not (lang_dir / 'ctc_topo_LG_uni.pt').exists():
            logging.info("Composing (ctc_topo, L, G)")
            first_phone_disambig_id = find_first_disambig_symbol(phones)
            first_word_disambig_id = find_first_disambig_symbol(words)
            # decoding_graph is the result of composing (ctc_topo, L, G)
            decoding_graph = compile_LG(
                L=L_disambig,
                G=G,
                ctc_topo=ctc_topo,
                labels_disambig_id_start=first_phone_disambig_id,
                aux_labels_disambig_id_start=first_word_disambig_id)
            torch.save(decoding_graph.as_dict(),
                       lang_dir / 'ctc_topo_LG_uni.pt')
        else:
            logging.info("Loading pre-compiled ctc_topo_LG")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lang_dir / 'ctc_topo_LG_uni.pt'))

        assert hasattr(decoding_graph, 'phones')

        self.decoding_graph = decoding_graph

    def compile(self, texts: Iterable[str],
                P: k2.Fsa) -> Tuple[k2.Fsa, k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
        Returns:
          A tuple (num_graph, den_graph, decoding_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.
              It is the result of compose(ctc_topo, P, L, transcript)

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph`.
              It is the result of compose(ctc_topo, P).

            - decoding_graph: It is the result of compose(ctc_topo, L, G)
              Note that it is a single Fsa, not an FsaVec.
        '''
        assert P.is_cpu()

        ctc_topo_P = k2.intersect(self.ctc_topo_inv, P).invert_()
        ctc_topo_P = k2.connect(ctc_topo_P)
        ctc_topo_P = k2.arc_sort(ctc_topo_P)

        num_graphs = k2.create_fsa_vec(
            [self.compile_one_and_cache(text) for text in texts])

        num = k2.compose(ctc_topo_P, num_graphs, inner_labels='phones')
        num = k2.connect(num)
        num = k2.arc_sort(num)

        den = k2.create_fsa_vec([ctc_topo_P.detach()] * len(texts))

        return num, den, self.decoding_graph

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
