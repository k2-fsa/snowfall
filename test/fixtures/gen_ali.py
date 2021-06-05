#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

# This file generates sample alignments for testing

from typing import Dict
from typing import List

import torch


def generate_alignment_1() -> Dict[str, Dict[str, List[int]]]:
    ans = {}
    ans['utt1'] = {}
    ans['utt1']['phone_label'] = [1, 2, 3, 5]
    ans['utt1']['ilabel'] = [10, 2, 3, 1, 6]

    ans['utt2'] = {}
    ans['utt2']['phone_label'] = [5, 3, 2, 1]
    ans['utt2']['ilabel'] = [6, 8, 2]

    return ans


def generate_alignment_2() -> Dict[str, Dict[str, List[int]]]:
    ans = {}
    ans['utt1'] = {}
    ans['utt1']['phone_label'] = [3, 3, 5]
    ans['utt1']['ilabel'] = [3, 2, 6]

    ans['utt2'] = {}
    ans['utt2']['phone_label'] = [2, 5, 2]
    ans['utt2']['ilabel'] = [6, 8, 1, 3]

    return ans


def generate_alignments():
    ref = generate_alignment_1()
    hyp = generate_alignment_2()

    torch.save(ref, 'ref.pt')
    torch.save(hyp, 'hyp.pt')


if __name__ == '__main__':
    generate_alignments()
