# Copyright (c)  2021  Xiaomi Corp.       (authors: Fangjun Kuang)

from pathlib import Path
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

import k2


def read_mapping(filename: Union[str, Path]) -> Dict[str, int]:
    '''Read a file that contains ID mappings.

    Each line in the file contains two fields separated by spaces.
    The first field is a token and the second is its integer ID.

    An example file may look like the following::

        a 1
        b 2
        hello 3

    Args:
      filename:
       Filename containing the mapping.
    Returns:
      Return a dict that maps a token to an integer.
    '''
    filename = Path(filename)
    assert filename.is_file(), f'{filename} is not a file'

    ans: Dict[str, int] = dict()
    seen: Set[int] = set()

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue  # skip empty lines

            splits = line.split()
            assert len(splits) == 2, \
                    f"Invalid line '{line}'.\n" \
                    'Each line should contain exactly two columns'

            key = splits[0]
            value = int(splits[1])
            assert key not in ans, \
                    f"Duplicate key '{key}' in line '{line}'"

            assert value not in seen, \
                    f"Duplicate ID '{value}' in line '{line}'"
            ans[key] = value
            seen.add(value)
    return ans


def convert_tokens_to_ids(tokens: List[str],
                          mapping: Dict[str, int]) -> List[int]:
    '''Convert a list of tokens to its corresponding IDs.

    Caution:
      We require that there are no OOVs. That is, every token
      present in `tokens` has a corresponding ID in `mapping`.

    Args:
      tokens:
        A list of str representing tokens.
      mapping:
        A map that maps a token to an integer.
    Returns:
      A list of integers that are the IDs of the input tokens.
    '''
    ans = []
    for t in tokens:
        assert t in mapping, f"token '{t}' does not have an ID"
        ans.append(mapping[t])
    return ans


def convert_lexicon_to_mappings(
        filename: Union[str, Path]
) -> Tuple[Dict[str, int], Dict[str, int]]:  # noqa
    '''Generate IDs for tokens from a lexicon file.

    Each line in the lexicon consists of spaces separated columns.
    The first column is the word and the remaining columns are its
    word pieces. We require that each word has a unique decomposition
    into word pieces.

    Args:
      filename:
        The lexicon file.
    Returns:
      Return a tuple containing two mappings:
        - The first dict maps a word to an ID
        - The second dict maps a word piece to an ID
    '''
    filename = Path(filename)
    assert filename.is_file(), f'File {filename} is not a file'

    words: Set[str] = set()
    pieces: Set[str] = set()

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue  # skip empty lines
            splits = line.split()
            assert len(splits) >= 2, \
                    f"Invalid line '{line}'.' \
                    'Expecting at least two columns"

            assert splits[0] not in words, "'Duplicate word '{splits[0]}'"
            words.add(splits[0])

            for p in splits[1:]:
                pieces.add(p)

    words = list(words)
    pieces = list(pieces)
    words.sort()
    pieces.sort()

    word2id: Dict[str, int] = dict()
    piece2id: Dict[str, int] = dict()

    for i, w in enumerate(words):
        word2id[w] = i

    for i, p in enumerate(pieces):
        piece2id[p] = i

    return word2id, piece2id


def read_lexicon(lexicon_filename: Union[Path, str]) -> Dict[str, List[str]]:
    '''Read a lexicon file.

    Each line in the lexicon consists of spaces separated columns.
    The first column is the word and the remaining columns are the
    corresponding word pieces.

    Args:
      lexicon_filename:
        Path to the lexicon.
    Returns:
      Return a dict mapping a word to its word pieces.
    '''
    lexicon_filename = Path(lexicon_filename)
    assert lexicon_filename.is_file(), f'File {lexicon_filename} is not a file'

    ans: Dict[str, List[str]] = dict()

    with open(lexicon_filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue  # skip empty lines

            splits = line.split()
            assert len(splits) >= 2, \
                    f"Invalid line '{line}'" \
                    'Expected a line with at least two fields'
            word = splits[0]

            assert word not in ans, \
                    f"Duplicate word '{word}' in line '{line}'"
            ans[word] = splits[1:]
    return ans


def create_ragged_lexicon(lexicon: Dict[str, List[str]],
                          word2id: Dict[str, int],
                          piece2id: Dict[str, int]) -> k2.RaggedInt:
    '''
    Args:
      lexicon:
        A dict that maps a word to word pieces.
      word2id:
        A dict that maps a word to an ID.

        CAUTION:
          We require that word IDs are contiguous. For instance, if
          there are 3 words, then the word IDs are 0, 1, and 2.
      piece2id:
        A dict that maps a word piece to an ID.
    '''
    # First, check that word IDs are contiguous
    id2word = {i: w for w, i in word2id.items()}
    ids = list(id2word.keys())
    ids.sort()
    # we assume that word IDs are contiguous
    expected_ids = list(range(ids[-1] + 1))
    assert ids == expected_ids

    values = []
    for i in ids:
        word = id2word[i]
        pieces = lexicon[word]
        pieces_id = convert_tokens_to_ids(pieces, piece2id)
        values.append(pieces_id)

    return k2.create_ragged2(values)
