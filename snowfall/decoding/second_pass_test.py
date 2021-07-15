#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from snowfall.decoding.second_pass import Nbest

import k2
import torch


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


def test_top_k():
    fsa0 = k2.Fsa.from_str('''
        0 1 -1 0
        1
    ''')
    fsas = [fsa0.clone() for i in range(10)]
    fsa_vec = k2.create_fsa_vec(fsas)
    fsa_vec.scores = torch.tensor([3, 0, 1, 5, 4, 2, 8, 1, 9, 6],
                                  dtype=torch.float)
    #    0 1   2 3 4   5 6 7 8 9
    # [ [3 0] [1 5 4] [2 8 1 9 6]
    shape = k2.RaggedShape('[ [x x] [x x x] [x x x x x] ]')
    nbest = Nbest(fsa_vec, shape)

    # top_k: k is 1
    nbest1 = nbest.top_k(1)
    expected_fsa = k2.create_fsa_vec([fsa_vec[0], fsa_vec[3], fsa_vec[8]])
    assert str(nbest1.fsa) == str(expected_fsa)

    expected_shape = k2.RaggedShape('[ [x] [x] [x] ]')
    assert nbest1.shape == expected_shape

    # top_k: k is 2
    nbest2 = nbest.top_k(2)
    expected_fsa = k2.create_fsa_vec([
        fsa_vec[0], fsa_vec[1], fsa_vec[3], fsa_vec[4], fsa_vec[8], fsa_vec[6]
    ])
    assert str(nbest2.fsa) == str(expected_fsa)

    expected_shape = k2.RaggedShape('[ [x x] [x x] [x x] ]')
    assert nbest2.shape == expected_shape

    # top_k: k is 3
    nbest3 = nbest.top_k(3)
    expected_fsa = k2.create_fsa_vec([
        fsa_vec[0], fsa_vec[1], fsa_vec[1], fsa_vec[3], fsa_vec[4], fsa_vec[2],
        fsa_vec[8], fsa_vec[6], fsa_vec[9]
    ])
    assert str(nbest3.fsa) == str(expected_fsa)

    expected_shape = k2.RaggedShape('[ [x x x] [x x x] [x x x] ]')
    assert nbest3.shape == expected_shape

    # top_k: k is 4
    nbest4 = nbest.top_k(4)
    expected_fsa = k2.create_fsa_vec([
        fsa_vec[0], fsa_vec[1], fsa_vec[1], fsa_vec[1], fsa_vec[3], fsa_vec[4],
        fsa_vec[2], fsa_vec[2], fsa_vec[8], fsa_vec[6], fsa_vec[9], fsa_vec[5]
    ])
    assert str(nbest4.fsa) == str(expected_fsa)

    expected_shape = k2.RaggedShape('[ [x x x x] [x x x x] [x x x x] ]')
    assert nbest4.shape == expected_shape
