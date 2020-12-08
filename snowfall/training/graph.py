from functools import lru_cache
from typing import Iterable, Dict, Optional

import k2
from k2 import Fsa, SymbolTable


class TrainingGraphCompiler:
    def __init__(self, L: Fsa, vocab: SymbolTable, oov: str = '<UNK>'):
        self.L = L
        self.vocab = vocab
        self.oov = oov
        self.cache: Dict[str, Fsa] = {}

    def compile(self, texts: Iterable[str]) -> Fsa:
        decoding_graphs = k2.union(k2.create_fsa_vec([self.compile_one_and_cache(text) for text in texts]))
        decoding_graphs.scores.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=50000)
    def compile_one_and_cache(self, text: str) -> Fsa:
        tokens = (token if token in self.vocab._sym2id else self.oov for token in text.split(' '))
        word_ids = [self.vocab.get(token) for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        decoding_graph = k2.intersect(fsa, self.L).invert_()
        decoding_graph = k2.add_epsilon_self_loops(decoding_graph)
        return decoding_graph


def create_decoding_graph(
        texts: Iterable[str],
        L: Fsa,
        symbols: SymbolTable,
        oov: str = '<UNK>'
) -> Fsa:
    word_ids_list = []
    for text in texts:
        filter_text = [token if token in symbols._sym2id else oov for token in text.split(' ')]
        word_ids = [symbols.get(i) for i in filter_text]
        word_ids_list.append(word_ids)
    fsa = k2.linear_fsa(word_ids_list)
    decoding_graph = k2.intersect(fsa, L).invert_()
    decoding_graph = k2.add_epsilon_self_loops(decoding_graph)
    return decoding_graph

