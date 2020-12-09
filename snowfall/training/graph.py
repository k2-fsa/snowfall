from functools import lru_cache
from typing import Iterable

import k2
from k2 import Fsa, SymbolTable


class TrainingGraphCompiler:
    """
    Compiles and caches the supervision FSAs created for text phrases.
    The Fsas are created and stored on the CPU and the user is expected to transfer
    them to GPU themselves (that might change at some point).

    Args:
        L_inv:
            An ``Fsa`` that represents the inverted lexicon (L), i.e. has words as ``symbols``
            and phones as ``aux_symbols``.
        vocab:
            The ``SymbolTable`` that corresponds to ``L_inv``.
        oov:
            A string token that's used to replace the words not present in ``vocab``.
            Has to be present in ``vocab``.

    """

    def __init__(self, L_inv: Fsa, vocab: SymbolTable, oov: str = '<UNK>'):
        self.L_inv = L_inv
        self.vocab = vocab
        self.oov = oov
        assert self.oov in self.vocab._sym2id, f"The OOV symbol \"{oov}\" is missing from the symbol table."

    def compile(self, texts: Iterable[str]) -> Fsa:
        decoding_graphs = k2.create_fsa_vec([self.compile_one_and_cache(text) for text in texts])
        decoding_graphs.requires_grad_(False)  # make sure the gradient is not accumulated
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def compile_one_and_cache(self, text: str) -> Fsa:
        tokens = (token if token in self.vocab._sym2id else self.oov for token in text.split(' '))
        word_ids = [self.vocab.get(token) for token in tokens]
        fsa = k2.linear_fsa(word_ids)
        decoding_graph = k2.connect(k2.intersect(fsa, self.L_inv)).invert_()
        decoding_graph = k2.add_epsilon_self_loops(decoding_graph)
        return decoding_graph
