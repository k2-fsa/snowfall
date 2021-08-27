#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
'''
Convert a transcript file to a corpus for LM training with
the help of a lexicon. If the lexicon contains phones, the resulting
LM will be a phone LM; If the lexicon contains word pieces,
the resulting LM will be a word piece LM.

If a word has multiple pronunciations, only the first one is used.

If the input transcript is:

    hello zoo world hello
    world zoo
    foo zoo world hellO

and if the lexicon is

    <UNK> SPN
    hello h e l l o 2
    hello h e l l o
    world w o r l d
    zoo z o o

Then the output is

    h e l l o 2 z o o w o r l d h e l l o 2
    w o r l d z o o
    SPN z o o w o r l d SPN
'''

from pathlib import Path
from typing import Dict

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript',
                        type=str,
                        help='The input transcript file.'
                        'We assume that the transcript file consists of '
                        'lines. Each line consists of space separated words.')
    parser.add_argument('--lexicon', type=str, help='The input lexicon file.')
    parser.add_argument('--oov',
                        type=str,
                        default='<UNK>',
                        help='The OOV word.')

    return parser.parse_args()


def read_lexicon(filename: str) -> Dict[str, str]:
    '''
    Args:
      filename:
        Filename to the lexicon. Each line in the lexicon
        has the following format:

            word p1 p2 p3

        where the first field is a word and the remaining fields
        are the pronunciations of the word. Fields are separated
        by spaces.
    Returns:
      Return a dict whose keys are words and values are the pronunciations.
    '''
    ans = dict()
    with open(filename) as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                # skip empty lines
                continue

            fields = line.split()
            assert len(fields) >= 2

            word = fields[0]
            pron = ' '.join(fields[1:])

            if word not in ans:
                # In case a word has multiple pronunciations,
                # we only use the first one
                ans[word] = pron
    return ans


def process_line(lexicon: Dict[str, str], line: str, oov_pron: str) -> None:
    '''
    Args:
      lexicon:
        A dict containing pronunciations. Its keys are words and values
        are pronunciations.
      line:
        A line of transcript consisting of space separated words.
      oov_pron:
        The pronunciation of the oov word if a word in line is not present
        in the lexicon.
    Returns:
      Return None.
    '''
    words = line.strip().split()
    for i, w in enumerate(words):
        pron = lexicon.get(w, oov_pron)
        print(pron, end=' ')
        if i == len(words) - 1:
            # end of the line, prints a new line
            print()


def main():
    args = get_args()
    assert Path(args.lexicon).is_file()
    assert Path(args.transcript).is_file()
    assert len(args.oov) > 0

    lexicon = read_lexicon(args.lexicon)
    assert args.oov in lexicon

    oov_pron = lexicon[args.oov]

    with open(args.transcript) as f:
        for line in f:
            process_line(lexicon=lexicon, line=line, oov_pron=oov_pron)


if __name__ == '__main__':
    main()
