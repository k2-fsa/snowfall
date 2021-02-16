#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
# Apache 2.0

import k2
import logging
import numpy as np
import os
import argparse
import torch
from k2 import Fsa, SymbolTable
from kaldialign import edit_distance
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from lhotse import CutSet
from lhotse.dataset import K2SpeechRecognitionDataset, SingleCutSampler
from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.common import get_texts
from snowfall.common import find_first_disambig_symbol
from snowfall.decoding.graph import compile_LG
from snowfall.models import AcousticModel
from snowfall.models.tdnn_lstm import TdnnLstm1c
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols


def decode(dataloader: torch.utils.data.DataLoader, model: AcousticModel,
           device: Union[str, torch.device], LG: Fsa, symbols: SymbolTable):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = []  # a list of pair (ref_words, hyp_words)
    for batch_idx, batch in enumerate(dataloader):
        feature = batch['features']
        supervisions = batch['supervisions']
        supervision_segments = torch.stack(
            (supervisions['sequence_idx'],
             torch.floor_divide(supervisions['start_frame'],
                                model.subsampling_factor),
             torch.floor_divide(supervisions['num_frames'],
                                model.subsampling_factor)), 1).to(torch.int32)
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        texts = supervisions['text']
        assert feature.ndim == 3

        feature = feature.to(device)
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        with torch.no_grad():
            nnet_output, _ = model(feature)
        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2,
                                          1)  # now nnet_output is [N, T, C]

        #  blank_bias = -3.0
        #  nnet_output[:, :, 0] += blank_bias

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
        # assert LG.is_cuda()
        assert LG.device == nnet_output.device, \
            f"Check failed: LG.device ({LG.device}) == nnet_output.device ({nnet_output.device})"
        # TODO(haowen): with a small `beam`, we may get empty `target_graph`,
        # thus `tot_scores` will be `inf`. Definitely we need to handle this later.
        lattices = k2.intersect_dense_pruned(LG, dense_fsa_vec, 20.0, 7.0, 30,
                                             10000)

        # lattices = k2.intersect_dense(LG, dense_fsa_vec, 10.0)
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        assert best_paths.shape[0] == len(texts)
        hyps = get_texts(best_paths, indices)
        assert len(hyps) == len(texts)

        for i in range(len(texts)):
            hyp_words = [symbols.get(x) for x in hyps[i]]
            ref_words = texts[i].split(' ')
            results.append((ref_words, hyp_words))

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))

        num_cuts += len(texts)

    return results


def print_transition_probabilities(P: k2.Fsa, phone_symbol_table: SymbolTable,
                                   phone_ids: List[int], filename: str):
    '''Print the transition probabilities of a phone LM.

    Args:
      P:
        A bigram phone LM.
      phone_symbol_table:
        The phone symbol table.
      phone_ids:
        A list of phone ids
      filename:
        Filename to save the printed result.
    '''
    num_phones = len(phone_ids)
    table = np.zeros((num_phones + 1, num_phones + 2))
    table[:, 0] = 0
    table[0, -1] = 0 # the start state has no arcs to the final state
    assert P.arcs.dim0() == num_phones + 2
    arcs = P.arcs.values()[:, :3]
    probability = P.scores.exp().tolist()

    assert arcs.shape[0] - num_phones == num_phones * (num_phones + 1)
    for i, arc in enumerate(arcs.tolist()):
        src_state, dest_state, label = arc[0], arc[1], arc[2]
        prob = probability[i]
        if label != -1:
            assert label == dest_state
        else:
            assert dest_state == num_phones + 1
        table[src_state][dest_state] = prob

    try:
        from prettytable import PrettyTable
    except ImportError:
        print('Please run `pip install prettytable`. Skip printing')
        return

    x = PrettyTable()

    field_names = ['source']
    field_names.append('sum')
    for i in phone_ids:
        field_names.append(phone_symbol_table[i])
    field_names.append('final')

    x.field_names = field_names

    for row in range(num_phones + 1):
        this_row = []
        if row == 0:
            this_row.append('start')
        else:
            this_row.append(phone_symbol_table[row])
        this_row.append('{:.6f}'.format(table[row, 1:].sum()))
        for col in range(1, num_phones + 2):
            this_row.append('{:.6f}'.format(table[row, col]))
        x.add_row(this_row)
    with open(filename, 'w') as f:
        f.write(str(x))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help="Decoding epoch.")
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100000,
        help="Maximum number of feature frames in a single batch.") 
    parser.add_argument(
        '--xent-scale',
        type=float,
        default=0.1,
        help="Cross entropy loss scale.")
    parser.add_argument(
        '--smooth-scale',
        type=float,
        default=0.1,
        help="Phone distribution smoothing loss scale.")
    return parser


def main():
    args = get_parser().parse_args()
    epoch = args.epoch
    max_frames = args.max_frames
    xent_scale = args.xent_scale
    smooth_scale = args.smooth_scale

    exp_dir = Path('exp-lstm-adam-mmi-xent-musan-' + 'xent-scale-' + str(xent_scale) + '-smooth-scale-' + str(smooth_scale) )
    setup_logger('{}/log/log-decode'.format(exp_dir), log_level='debug')

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')

    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)

    phone_ids_with_blank = [0] + phone_ids
    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

    logging.debug("About to load model")
    # Note: Use "export CUDA_VISIBLE_DEVICES=N" to setup device id to N
    # device = torch.device('cuda', 1)
    device = torch.device('cuda')
    model = TdnnLstm1c(num_features=40,
                       num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
                       subsampling_factor=3)
    model.P_scores = torch.nn.Parameter(P.scores.clone(), requires_grad=False)

    checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
    load_checkpoint(checkpoint, model)
    model.to(device)
    model.eval()

    assert P.requires_grad is False
    P.scores = model.P_scores.cpu()
    print_transition_probabilities(P, phone_symbol_table, phone_ids, filename='model_P_scores.txt')

    P.set_scores_stochastic_(model.P_scores)
    print_transition_probabilities(P, phone_symbol_table, phone_ids, filename='P_scores.txt')

    if not os.path.exists(lang_dir / 'LG.pt'):
        logging.debug("Loading L_disambig.fst.txt")
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        logging.debug("Loading G.fst.txt")
        with open(lang_dir / 'G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        LG = compile_LG(L=L,
                        G=G,
                        ctc_topo=ctc_topo,
                        labels_disambig_id_start=first_phone_disambig_id,
                        aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(LG.as_dict(), lang_dir / 'LG.pt')
    else:
        logging.debug("Loading pre-compiled LG")
        d = torch.load(lang_dir / 'LG.pt')
        LG = k2.Fsa.from_dict(d)

    # load dataset
    feature_dir = Path('exp/data')
    logging.debug("About to get test cuts")
    cuts_test = CutSet.from_json(feature_dir / 'cuts_test-clean.json.gz')

    logging.info("About to create test dataset")
    test = K2SpeechRecognitionDataset(cuts_test)
    sampler = SingleCutSampler(cuts_test, max_frames=max_frames)
    logging.info("About to create test dataloader")
    test_dl = torch.utils.data.DataLoader(test, batch_size=None, sampler=sampler, num_workers=1)

    #  if not torch.cuda.is_available():
    #  logging.error('No GPU detected!')
    #  sys.exit(-1)

    logging.debug("convert LG to device")
    LG = LG.to(device)
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)
    LG.requires_grad_(False)
    logging.debug("About to decode")
    results = decode(dataloader=test_dl,
                     model=model,
                     device=device,
                     LG=LG,
                     symbols=symbol_table)
    s = ''
    for ref, hyp in results:
        s += f'ref={ref}\n'
        s += f'hyp={hyp}\n'
    logging.info(s)
    # compute WER
    dists = [edit_distance(r, h) for r, h in results]
    errors = {
        key: sum(dist[key] for dist in dists)
        for key in ['sub', 'ins', 'del', 'total']
    }
    total_words = sum(len(ref) for ref, _ in results)
    # Print Kaldi-like message:
    # %WER 8.20 [ 4459 / 54402, 695 ins, 427 del, 3337 sub ]
    logging.info(
        f'%WER {errors["total"] / total_words:.2%} '
        f'[{errors["total"]} / {total_words}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
    )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
