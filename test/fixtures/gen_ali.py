#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

# This file generates sample alignments for testing

from typing import Dict
from typing import List

import k2
import torch

from snowfall.tools.ali import Alignment


def generate_alignment_1() -> Dict[str, Alignment]:
    ans = {}

    ali = Alignment({})
    ali.value['phone'] = [1, 2, 3, 5]
    ali.value['ilabel'] = [10, 2, 3, 1, 6]
    ans['utt1'] = ali

    ali = Alignment({})
    ali.value['phone'] = [5, 3, 2, 1]
    ali.value['ilabel'] = [6, 8, 2]
    ans['utt2'] = ali

    ali = Alignment({})
    ali.value['phone'] = [5, 1, 3, 2]
    ali.value['ilabel'] = [9, 1, 3, 5]
    ans['utt3'] = ali

    return ans


def generate_alignment_2() -> Dict[str, Alignment]:
    ans = {}

    ali = Alignment({})
    ali.value['phone'] = [3, 3, 5]
    ali.value['ilabel'] = [3, 2, 6]
    ans['utt1'] = ali

    ali = Alignment({})
    ali.value['phone'] = [2, 5, 2]
    ali.value['ilabel'] = [6, 8, 1, 3]
    ans['utt2'] = ali

    ali = Alignment({})
    ali.value['phone'] = [2, 5, 5, 1, 3]
    ali.value['ilabel'] = [2, 2, 3, 5, 6]
    ans['utt4'] = ali

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
