#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from pathlib import Path
from typing import List

import argparse
import sentencepiece as spm


def read_words(words_txt: str, excluded=['<eps>', '<UNK>']) -> List[str]:
    '''Read words_txt and return a list of words.
    The file words_txt has the following format:
        <word> <id>
    That is, every line has two fields. This function
    extracts the first field.
    Args:
      words_txt:
        Filename of words.txt.
      excluded:
        words in this list are not returned.
    Returns:
      Return a list of words.
    '''
    ans = []
    with open(words_txt, 'r', encoding='latin-1') as f:
        for line in f:
            word, _ = line.strip().split()
            if word not in excluded:
                ans.append(word)
    return ans


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file',
                        type=str,
                        help='Pre-trained BPE model file')

    parser.add_argument('--words-file', type=str, help='Path to words.txt')

    args = parser.parse_args()
    model_file = args.model_file
    words_txt = args.words_file
    assert Path(model_file).is_file(), f'{model_file} does not exist'
    assert Path(words_txt).is_file(), f'{words_txt} does not exist'

    words = read_words(words_txt)

    sp = spm.SentencePieceProcessor()
    sp.load(model_file)

    for word in words:
        pieces = sp.EncodeAsPieces(word.upper())
        print(word, ' '.join(pieces))

    print('<UNK>', '<UNK>')


if __name__ == '__main__':
    main()
