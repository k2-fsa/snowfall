from functools import lru_cache
from typing import Dict, Iterable, Optional

import k2
from k2 import Fsa, SymbolTable


class TrainingGraphCompiler:
    def __init__(self, L: Fsa, vocab: SymbolTable, oov: str = '<UNK>'):
        self.L = L
        self.vocab = vocab
        self.oov = oov

    def compile(self, texts: Iterable[str]) -> Fsa:
        decoding_graphs = k2.create_fsa_vec([self.compile_one_and_cache(text) for text in texts])
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
