#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from snowfall.decoding.second_pass import Nbest

import k2


def test_nbest_constructor():
    fsa = k2.Fsa.from_str('''
        0 1 -1 0.1
        1
    ''')

    fsa_vec = k2.create_fsa_vec([fsa, fsa, fsa])
    shape = k2.RaggedShape('[[x x] [x]]')
    print(shape.num_axes())

    nbest = Nbest(fsa_vec, shape)
    print(nbest)
