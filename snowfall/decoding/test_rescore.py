#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)
#

from pathlib import Path

cur_dir = Path(__file__).resolve().parent
snowfall_dir = cur_dir.parent.parent

import sys
sys.path.insert(0, f'{snowfall_dir}')
sys.path.insert(0, '/root/fangjun/open-source/k2/build/lib')
sys.path.insert(0, '/root/fangjun/open-source/k2/k2/python')

import k2
import torch

from snowfall.decoding.rescore import get_paths
from snowfall.decoding.rescore import get_word_fsas
from snowfall.decoding.rescore import rescore
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.models import Tdnn2aEmbedding


def generate_lats() -> k2.Fsa:
    '''Return an FsaVec that contains 2 seqs.
    '''
    s1 = '''
        0 1 1 0.1
        0 2 2 0.2
        1 3 0 0.3
        1 4 2 0.4
        1 6 1 0.45
        2 4 0 0.5
        2 5 5 0.6
        3 6 3 0.7
        4 6 0 0.8
        4 7 4 0.9
        5 7 1 1.0
        6 8 -1 0.0
        7 8 -1 0.0
        8
    '''
    fsa1 = k2.Fsa.from_str(s1)
    fsa1.phones = fsa1.labels.clone()
    fsa1.scores = torch.rand_like(fsa1.scores).log()
    fsa1.aux_labels = torch.arange(fsa1.phones.numel(), dtype=torch.int32) + 10
    fsa1.aux_labels[-2:] = -1

    s2 = '''
        0 1 1 0.1
        0 2 2 0.2
        1 3 3 0.3
        1 4 0 0.4
        2 3 6 0.5
        2 5 4 0.6
        3 4 1 0.7
        3 5 0 0.8
        4 6 -1 0.0
        5 6 -1 0.0
        6
    '''
    fsa2 = k2.Fsa.from_str(s2)
    fsa2.phones = fsa2.labels.clone()
    fsa2.scores = torch.rand_like(fsa2.scores).log()
    fsa2.aux_labels = torch.arange(fsa2.phones.numel(), dtype=torch.int32) + 20
    fsa2.aux_labels[-2:] = -1

    lats = k2.create_fsa_vec([fsa1, fsa2])
    return lats


def main() -> None:
    lats = generate_lats()
    max_phone_id = max(lats.phones)
    phone_ids_with_blank = list(range(max_phone_id + 1))

    N = lats.shape[0]
    T = 12
    C = len(phone_ids_with_blank)
    nnet_output = torch.rand(N, T, C).log()
    supervision_segments = torch.tensor([[0, 0, T], [0, 0, T - 2]],
                                        dtype=torch.int32)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    ctc_topo = build_ctc_topo(phone_ids_with_blank)

    num_embedding_features = len(phone_ids_with_blank) + max_phone_id + 3
    second_pass_model = Tdnn2aEmbedding(num_features=num_embedding_features,
                                        num_classes=len(phone_ids_with_blank))

    paths = get_paths(lats=lats, num_paths=3)
    word_fsas, seq_to_path_shape = get_word_fsas(lats, paths)
    replicated_lats = k2.index(lats, seq_to_path_shape.row_ids(1))
    word_lats = k2.compose(replicated_lats,
                           word_fsas,
                           treat_epsilons_specially=False)
    tot_scores_1st = word_lats.get_tot_scores(use_double_scores=True,
                                              log_semiring=True)

    best_paths = rescore(lats=lats,
                         paths=paths,
                         tot_scores_1st=tot_scores_1st,
                         seq_to_path_shape=seq_to_path_shape,
                         ctc_topo=ctc_topo,
                         decoding_graph=ctc_topo,
                         dense_fsa_vec=dense_fsa_vec,
                         second_pass_model=second_pass_model,
                         max_phone_id=max_phone_id)


if __name__ == '__main__':
    torch.manual_seed(20210312)
    main()
