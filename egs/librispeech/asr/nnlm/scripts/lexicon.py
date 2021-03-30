# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#

from pathlib import Path
from typing import Union

from util import create_ragged_lexicon
from util import read_lexicon
from util import read_mapping

import torch
import k2


class Lexicon(object):

    def __init__(self, lexicon_filename: Union[Path, str],
                 word2id_filename: Union[Path, str],
                 piece2id_filename: Union[Path, str]) -> None:
        '''
        Args:
          lexicon_filename:
            Path to the lexicon file. Each line in it consists of
            spaces separated columns. The first column is a word
            and the remaining columns are the word pieces of this word.
          word2id_filename:
            Path to the file that maps a word to an ID.
          piece2id_filename:
            Path to the file that maps a word piece to an ID.
        '''
        lexicon = read_lexicon(lexicon_filename)
        self.word2id = read_mapping(word2id_filename)
        piece2id = read_mapping(piece2id_filename)

        self.lexicon = create_ragged_lexicon(lexicon=lexicon,
                                             word2id=self.word2id,
                                             piece2id=piece2id)

    def word_seq_to_word_piece_seq(self, words: torch.Tensor) -> torch.Tensor:
        '''Convert a word sequence to a word piece seq.

        Args:
          words:
            A 1-D torch.Tensor of dtype torch.int32 containing word IDs.
          Returns:
            Return a 1-D torch.Tensor containing the IDs of the
            corresponding word pieces.
        '''
        assert words.ndim == 1
        assert words.dtype == torch.int32

        ans = k2.index(self.lexicon, words)
        return ans.values()
