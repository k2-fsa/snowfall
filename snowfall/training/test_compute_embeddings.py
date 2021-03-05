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

from snowfall.training.compute_embeddings import compute_embeddings
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols
from snowfall.training.mmi_mbr_graph import MmiMbrTrainingGraphCompiler

import torch
import k2


def main():
    torch.manual_seed(20210220)
    device = torch.device('cuda', 0)
    #  device = torch.device('cpu')

    d = '/root/fangjun/open-source/snowfall/egs/librispeech/asr/simple_v1'
    lang_dir = Path(f'{d}/data/lang_nosp')

    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')

    with open(lang_dir / 'L.fst.txt') as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False).to(device)
        L_inv = k2.arc_sort(L.invert_())

    with open(lang_dir / 'L_disambig.fst.txt') as f:
        L_disambig = k2.Fsa.from_openfst(f.read(), acceptor=False).to(device)
        L_disambig = k2.arc_sort(L_disambig)

    with open(lang_dir / 'G_uni.fst.txt') as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False).to(device)
        G = k2.arc_sort(G)

    graph_compiler = MmiMbrTrainingGraphCompiler(L_inv=L_inv,
                                                 L_disambig=L_disambig,
                                                 G=G,
                                                 device=device,
                                                 phones=phone_symbol_table,
                                                 words=word_symbol_table)
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids).to(device)
    scores = torch.randn_like(P.scores)
    P.set_scores_stochastic_(scores)

    texts = ['HELLO', 'PEOPLE']
    num_graph, den_graph, decoding_graph = graph_compiler.compile(texts, P)

    N = 2
    T = 1000
    C = len(phone_ids) + 1
    nnet_output = torch.rand(
        N, T, C,
        dtype=torch.float32).softmax(-1).log().to(device).requires_grad_(True)

    supervision_segments = torch.tensor([[0, 0, T], [1, 0, T - 10]],
                                        dtype=torch.int32)

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    num_lats = k2.intersect_dense(num_graph,
                                  dense_fsa_vec,
                                  10.0,
                                  seqframe_idx_name='seqframe_idx')

    mbr_lats = k2.intersect_dense_pruned(decoding_graph,
                                         dense_fsa_vec,
                                         20.0,
                                         7.0,
                                         30,
                                         10000,
                                         seqframe_idx_name='seqframe_idx')

    den_lats = k2.intersect_dense(den_graph, dense_fsa_vec, 10.0)

    print('-' * 10, 'den_lats', '-' * 10)
    den_padded_embeddings, den_len_per_path, den_path_to_seq, den_num_repeats = compute_embeddings(
        den_lats,
        graph_compiler.ctc_topo,
        dense_fsa_vec,
        max_phone_id=graph_compiler.max_phone_id,
        num_paths=10,
        debug=True)
    assert den_padded_embeddings.ndim == 3
    assert den_padded_embeddings.shape[1] == den_len_per_path.max()
    assert den_padded_embeddings.shape[2] == (dense_fsa_vec.scores.shape[1] -
                                              1 + graph_compiler.max_phone_id +
                                              2 + 1)
    assert den_padded_embeddings.shape[0] == den_len_per_path.shape[0]
    assert den_padded_embeddings.shape[0] == den_path_to_seq.shape[0]
    assert 0 <= den_path_to_seq.max() < supervision_segments.shape[0]
    assert den_len_per_path.ndim == 1
    assert den_len_per_path.dtype == torch.int32
    assert den_path_to_seq.ndim == 1
    assert den_path_to_seq.dtype == torch.int32
    assert den_len_per_path.device.type == 'cpu'
    assert den_path_to_seq.device == device

    assert den_padded_embeddings.requires_grad is True
    assert den_padded_embeddings.dtype == torch.float32

    print('den', den_num_repeats)

    print('-' * 10, 'mbr_lats', '-' * 10)
    mbr_padded_embeddings, mbr_len_per_path, mbr_path_to_seq, mbr_num_repeats = compute_embeddings(
        mbr_lats,
        graph_compiler.ctc_topo,
        dense_fsa_vec,
        max_phone_id=graph_compiler.max_phone_id,
        num_paths=10,
        debug=True)

    assert mbr_padded_embeddings.ndim == 3
    assert mbr_padded_embeddings.shape[1] == mbr_len_per_path.max()
    assert mbr_padded_embeddings.shape[2] == (dense_fsa_vec.scores.shape[1] -
                                              1 + graph_compiler.max_phone_id +
                                              2 + 1)

    assert mbr_padded_embeddings.shape[0] == mbr_len_per_path.shape[0]
    assert mbr_padded_embeddings.shape[0] == mbr_path_to_seq.shape[0]
    assert 0 <= mbr_path_to_seq.max() < supervision_segments.shape[0]
    assert mbr_len_per_path.ndim == 1
    assert mbr_len_per_path.dtype == torch.int32
    assert mbr_path_to_seq.ndim == 1
    assert mbr_path_to_seq.dtype == torch.int32
    assert mbr_len_per_path.device.type == 'cpu'
    assert mbr_path_to_seq.device == device

    assert mbr_padded_embeddings.requires_grad is True
    assert mbr_padded_embeddings.dtype == torch.float32

    print('mbr', mbr_num_repeats)
    print(mbr_padded_embeddings.sum())


if __name__ == '__main__':
    main()
