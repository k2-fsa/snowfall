#!/usr/bin/env python3

from pathlib import Path

cur_dir = Path(__file__).resolve().parent
snowfall_dir = cur_dir.parent.parent

import sys
sys.path.insert(0, f'{snowfall_dir}')

import torch

from snowfall.models import Tdnn2aEmbedding


def test_tdnn2a_embedding_case_1():
    N = 5
    num_features = 3
    num_classes = 8
    model = Tdnn2aEmbedding(num_features=num_features, num_classes=num_classes)
    for T in range(1, 50):
        feats = torch.rand(N, num_features, T)
        pred = model(feats)
        assert pred.shape == (N, num_classes, T)


def test_tdnn2a_embedding_case_2():
    num_features = 3
    num_classes = 8
    model = Tdnn2aEmbedding(num_features=num_features, num_classes=num_classes)

    feature_length = [3, 8, 10, 6]
    N = len(feature_length)
    max_T = max(feature_length)
    features = []
    for T in feature_length:
        feature = torch.rand(T, num_features)
        features.append(feature)

    padded_features = torch.nn.utils.rnn.pad_sequence(features,
                                                      batch_first=True)
    assert padded_features.shape == (N, max_T, num_features)
    # from (N, T, C) to (N, C, T)
    padded_features = padded_features.permute(0, 2, 1)
    predicated = model(padded_features)
    assert predicated.shape == (N, num_classes, max_T)


def main():
    test_tdnn2a_embedding_case_1()
    test_tdnn2a_embedding_case_2()


if __name__ == '__main__':
    main()
