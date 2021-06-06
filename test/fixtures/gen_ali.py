#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

# This file generates sample alignments for testing

from typing import Dict
from typing import List

import k2
import torch

from snowfall.tools.ali import Alignment


def generate_alignment_1() -> Dict[str, Dict[str, List[int]]]:
    ans = {}

    ans['utt1'] = []
    phones = Alignment('phone_label', [1, 2, 3, 5])
    ilabels = Alignment('ilabel', [10, 2, 3, 1, 6])
    ans['utt1'].append(phones)
    ans['utt1'].append(ilabels)

    ans['utt2'] = []
    phones = Alignment('phone_label', [5, 3, 2, 1])
    ilabels = Alignment('ilabel', [6, 8, 2])
    ans['utt2'].append(phones)
    ans['utt2'].append(ilabels)

    return ans


def generate_alignment_2() -> Dict[str, Dict[str, List[int]]]:
    ans = {}

    ans['utt1'] = []
    phones = Alignment('phone_label', [3, 3, 5])
    ilabels = Alignment('ilabel', [3, 2, 6])
    ans['utt1'].append(phones)
    ans['utt1'].append(ilabels)

    ans['utt2'] = []
    phones = Alignment('phone_label', [2, 5, 2])
    ilabels = Alignment('ilabel', [6, 8, 1, 3])
    ans['utt2'].append(phones)
    ans['utt2'].append(ilabels)

    return ans


def generate_symbol_table() -> k2.SymbolTable:
    s = '''
    one 1
    two 2
    three 3
    four 4
    five 5
    six 6
    seven 7
    eight 8
    nine 9
    ten 10
    '''
    return k2.SymbolTable.from_str(s)


def generate_alignments():
    ref = generate_alignment_1()
    hyp = generate_alignment_2()

    torch.save(ref, 'ref.pt')
    torch.save(hyp, 'hyp.pt')

    sym = generate_symbol_table()
    sym.to_file('sym.txt')


if __name__ == '__main__':
    generate_alignments()
