# Copyright (c)  2021  Xiaomi Corporation (authors: Guo Liyong)
# Apache 2.0

from typing import List, Union
from pathlib import Path

import k2
import sentencepiece as spm


Pathlike = Union[str, Path]

class Numericalizer(object):
    def __init__(self, tokenizer, tokens_list):
        super().__init__()
        self.tokenizer=tokenizer
        self.tokens_list = tokens_list

    def EncodeAsIds(self, text: str) -> List[int]:
        tokens = self.tokenizer.EncodeAsPieces(text.upper())
        assert len(tokens) != 0
        tokenids = [self.tokens_list[token] for token in tokens]
        return tokenids

    @classmethod
    def build_numericalizer(cls, tokenizer_model_file: str, tokens_file: Pathlike):
        sp = spm.SentencePieceProcessor()
        if not isinstance(tokenizer_model_file, str):
            # sp.Load only support path in str format
            assert isinstance(tokenizer_model_file, Path)
            tokenizer_model_file = str(tokenizer_model_file)
        sp.Load(tokenizer_model_file)
        tokens_list = k2.SymbolTable.from_file(tokens_file)
        assert  sp.GetPieceSize() == len(tokens_list)
        return Numericalizer(sp, tokens_list)

