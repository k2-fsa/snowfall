#!/usr/bin/env python3

from pathlib import Path

import os
import tempfile

from lexicon import Lexicon
from util import convert_lexicon_to_mappings
from util import convert_tokens_to_ids
from util import read_lexicon
from util import read_mapping

import torch


def get_temp_filename() -> str:
    '''Return a temporary file.

    The caller is expected to remove the returned file.
    '''
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        name = tmp.name
        tmp.close()
    return name


def generate_mapping_file() -> str:
    '''Generate a temporary mapping file for testing.

    Caution:
      The caller is responsible to delete the returned file after using it.

    Returns:
      A temporary file that contains an example mapping.
    '''
    s = '''
        a 1
        b 2
        hello 3
    '''
    filename = get_temp_filename()
    with open(filename, 'w') as f:
        f.write(s)
    return filename


def generate_lexicon_file() -> str:
    '''Generate a temporary lexicon file for testing.

    Caution:
      The caller is responsible to delete the returned file after using it.

    Returns:
      A temporary file that contains an example lexicon.
    '''
    s = '''
        tom to m
        the the
        piper p ip er
        son so n
    '''
    filename = get_temp_filename()
    with open(filename, 'w') as f:
        f.write(s)
    return filename


def test_read_mapping_file():
    filename = generate_mapping_file()
    mapping = read_mapping(filename)
    os.remove(filename)
    assert mapping['a'] == 1
    assert mapping['b'] == 2
    assert mapping['hello'] == 3


def test_convert_tokens_to_ids():
    filename = generate_mapping_file()
    mapping = read_mapping(filename)
    os.remove(filename)

    tokens = ['b', 'a', 'a', 'hello', 'a', 'a', 'b']
    ids = convert_tokens_to_ids(tokens=tokens, mapping=mapping)
    assert ids == [2, 1, 1, 3, 1, 1, 2]


def test_convert_lexicon_to_mappings():
    filename = generate_lexicon_file()
    word2id, piece2id = convert_lexicon_to_mappings(filename)
    print(word2id)
    print(piece2id)
    os.remove(filename)


def test_read_lexicon():
    filename = generate_lexicon_file()
    lexicon = read_lexicon(filename)
    os.remove(filename)
    print(lexicon)


def test_lexicon():
    lexicon_filename = generate_lexicon_file()

    word2id, piece2id = convert_lexicon_to_mappings(lexicon_filename)

    word2id_filename = get_temp_filename()
    piece2id_filename = get_temp_filename()
    # piper: 0
    # son: 1
    # the 2
    # tome 3

    # er 0
    # ip 1
    # m 2
    # n 3
    # p 4
    # so 5
    # the 6
    # to 7

    with open(word2id_filename, 'w') as f:
        for w, i in word2id.items():
            f.write(f'{w} {i}\n')

    with open(piece2id_filename, 'w') as f:
        for p, i in piece2id.items():
            f.write(f'{p} {i}\n')

    lexicon = Lexicon(lexicon_filename, word2id_filename, piece2id_filename)
    words = ['the', 'son', 'tom', 'piper', 'the']
    word_ids = convert_tokens_to_ids(words, word2id)
    word_piece_ids = lexicon.word_seq_to_word_piece_seq(
        torch.tensor(word_ids, dtype=torch.int32))
    # the so n to m p ip er the
    #  6   5 3 7  2 4 1  0  6
    expected_word_piece_ids = torch.tensor([6, 5, 3, 7, 2, 4, 1, 0, 6])

    assert torch.all(torch.eq(word_piece_ids, expected_word_piece_ids))

    os.remove(lexicon_filename)
    os.remove(word2id_filename)
    os.remove(piece2id_filename)


if __name__ == '__main__':
    test_read_mapping_file()
    test_convert_tokens_to_ids()
    test_convert_lexicon_to_mappings()
    test_read_lexicon()
    test_lexicon()
