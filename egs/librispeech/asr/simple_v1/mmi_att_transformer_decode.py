#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import k2
import logging
import numpy as np
import os
import torch
from k2 import Fsa, SymbolTable
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


from snowfall.common import average_checkpoint, store_transcripts
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_texts
from snowfall.common import write_error_stats
from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.common import str2bool
from snowfall.data import LibriSpeechAsrDataModule
from snowfall.decoding.graph import compile_HLG
from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from snowfall.models import AcousticModel
from snowfall.models.transformer import Transformer
from snowfall.models.conformer import Conformer
from snowfall.models.contextnet import ContextNet
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols


def decode_one_batch(batch: Dict[str, Any],
                     model: AcousticModel,
                     HLG: k2.Fsa,
                     output_beam_size: float,
                     num_paths: int,
                     use_whole_lattice: bool,
                     G: Optional[k2.Fsa] = None)->Dict[str, List[List[int]]]:
    device = HLG.device
    feature = batch['inputs']
    assert feature.ndim == 3
    feature = feature.to(device)

    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]

    supervisions = batch['supervisions']

    nnet_output, _, _ = model(feature, supervisions)
    # nnet_output is [N, C, T]

    nnet_output = nnet_output.permute(0, 2, 1)
    # now nnet_output is [N, T, C]

    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)),
        1).to(torch.int32)

    supervision_segments = torch.clamp(supervision_segments, min=0)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

    if G is None:
        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        hyps = get_texts(best_paths, indices)
        return {'no_rescore': hyps}

    lm_scale_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    ans = dict()
    saved_scores = lattices.scores.clone()
    for lm_scale in lm_scale_list:
        lattices.scores = saved_scores.clone()
        best_paths = decode_with_lm_rescoring(
            lattices,
            G,
            num_paths=num_paths,
            use_whole_lattice=use_whole_lattice,
            lm_scale=lm_scale)

        hyps = get_texts(best_paths, indices)
        key = f'lm_scale_{lm_scale}'
        ans[key] = hyps
    return ans


@torch.no_grad()
def decode(dataloader: torch.utils.data.DataLoader, model: AcousticModel,
           HLG: Fsa, symbols: SymbolTable,
           num_paths: int, G: k2.Fsa, use_whole_lattice: bool, output_beam_size: float):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = defaultdict(list)
    for batch_idx, batch in enumerate(dataloader):
        texts = batch['supervisions']['text']

        hyps_dict = decode_one_batch(batch=batch,
                                     model=model,
                                     HLG=HLG,
                                     output_beam_size=output_beam_size,
                                     num_paths=num_paths,
                                     use_whole_lattice=use_whole_lattice,
                                     G=G)

        for key, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)

            for i in range(len(texts)):
                hyp_words = [symbols.get(x) for x in hyps[i]]
                ref_words = texts[i].split(' ')
                this_batch.append((ref_words, hyp_words))

            results[key].extend(this_batch)

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))

        num_cuts += len(texts)

    return results


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer", "contextnet"],
        help="Model type.")
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help="Decoding epoch.")
    parser.add_argument(
        '--avg',
        type=int,
        default=5,
        help="Number of checkpionts to average. Automatically select "
             "consecutive checkpoints before checkpoint specified by'--epoch'. ")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=4,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=256,
        help="Number of units in transformer attention layers.")
    parser.add_argument(
        '--output-beam-size',
        type=float,
        default=8,
        help='Output beam size. Used in k2.intersect_dense_pruned.'\
             'Choose a large value (e.g., 20), for 1-best decoding '\
             'and n-best rescoring. Choose a small value (e.g., 8) for ' \
             'rescoring with the whole lattice')
    parser.add_argument(
        '--use-lm-rescoring',
        type=str2bool,
        default=True,
        help='When enabled, it uses LM for rescoring')
    parser.add_argument(
        '--num-paths',
        type=int,
        default=-1,
        help='Number of paths for rescoring using n-best list.' \
             'If it is negative, then rescore with the whole lattice.'\
             'CAUTION: You have to reduce max_duration in case of CUDA OOM'
             )
    parser.add_argument(
        '--is-espnet-structure',
        type=str2bool,
        default=True,
        help='When enabled, the conformer will have the ' \
             'same structure like espnet')
    parser.add_argument(
        '--vgg-frontend',
        type=str2bool,
        default=True,
        help='When enabled, it uses vgg style network for subsampling')
    return parser


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    model_type = args.model_type
    epoch = args.epoch
    avg = args.avg
    att_rate = args.att_rate
    num_paths = args.num_paths
    use_lm_rescoring = args.use_lm_rescoring
    use_whole_lattice = False
    if use_lm_rescoring and num_paths < 1:
        # It doesn't make sense to use n-best list for rescoring
        # when n is less than 1
        use_whole_lattice = True

    output_beam_size = args.output_beam_size

    exp_dir = Path('exp-' + model_type + '-mmi-att-sa-vgg-normlayer')
    setup_logger('{}/log/log-decode'.format(exp_dir), log_level='debug')

    logging.info(f'output_beam_size: {output_beam_size}')

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

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    if model_type == "transformer":
        model = Transformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=args.vgg_fronted)
    elif model_type == "conformer":
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=args.vgg_frontend,
            is_espnet_structure=args.is_espnet_structure)
    elif model_type == "contextnet":
        model = ContextNet(
        num_features=80,
        num_classes=len(phone_ids) + 1) # +1 for the blank symbol
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not implemented")

    model.P_scores = torch.nn.Parameter(P.scores.clone(), requires_grad=False)

    if avg == 1:
        checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
        load_checkpoint(checkpoint, model)
    else:
        checkpoints = [os.path.join(exp_dir, 'epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                       range(epoch - avg, epoch)]
        average_checkpoint(checkpoints, model)

    model.to(device)
    model.eval()

    if not os.path.exists(lang_dir / 'HLG.pt'):
        logging.debug("Loading L_disambig.fst.txt")
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        logging.debug("Loading G.fst.txt")
        with open(lang_dir / 'G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        HLG = compile_HLG(L=L,
                         G=G,
                         H=ctc_topo,
                         labels_disambig_id_start=first_phone_disambig_id,
                         aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')
    else:
        logging.debug("Loading pre-compiled HLG")
        d = torch.load(lang_dir / 'HLG.pt')
        HLG = k2.Fsa.from_dict(d)

    if use_lm_rescoring:
        if use_whole_lattice:
            logging.info('Rescoring with the whole lattice')
        else:
            logging.info(f'Rescoring with n-best list, n is {num_paths}')
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        if not os.path.exists(lang_dir / 'G_4_gram.pt'):
            logging.debug('Loading G_4_gram.fst.txt')
            with open(lang_dir / 'G_4_gram.fst.txt') as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION(fangjun): The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                G = k2.create_fsa_vec([G]).to(device)
                G = k2.arc_sort(G)
                torch.save(G.as_dict(), lang_dir / 'G_4_gram.pt')
        else:
            logging.debug('Loading pre-compiled G_4_gram.pt')
            d = torch.load(lang_dir / 'G_4_gram.pt')
            G = k2.Fsa.from_dict(d).to(device)

        if use_whole_lattice:
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)
    else:
        logging.debug('Decoding without LM rescoring')
        G = None

    logging.debug("convert HLG to device")
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    if not hasattr(HLG, 'lm_scores'):
        HLG.lm_scores = HLG.scores.clone()

    # load dataset
    librispeech = LibriSpeechAsrDataModule(args)
    test_sets = ['test-clean', 'test-other']
    #  test_sets = ['test-other']
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        logging.info(f'* DECODING: {test_set}')

        test_set_wers = dict()
        results_dict = decode(dataloader=test_dl,
                              model=model,
                              HLG=HLG,
                              symbols=symbol_table,
                              num_paths=num_paths,
                              G=G,
                              use_whole_lattice=use_whole_lattice,
                              output_beam_size=output_beam_size)

        for key, results in results_dict.items():
            recog_path = exp_dir / f'recogs-{test_set}-{key}.txt'
            store_transcripts(path=recog_path, texts=results)
            logging.info(f'The transcripts are stored in {recog_path}')

            # The following prints out WERs, per-word error statistics and aligned
            # ref/hyp pairs.
            errs_filename = exp_dir / f'errs-{test_set}-{key}.txt'
            with open(errs_filename, 'w') as f:
                wer = write_error_stats(f, f'{test_set}-{key}', results)
                test_set_wers[key] = wer

            logging.info('Wrote detailed error stats to {}'.format(errs_filename))

        test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
        errs_info = exp_dir / f'wer-summary-{test_set}.txt'
        with open(errs_info, 'w') as f:
            print('settings\tWER', file=f)
            for key, val in test_set_wers:
                print('{}\t{}'.format(key, val), file=f)

        s = '\nFor {}, WER of different settings are:\n'.format(test_set)
        note = '\tbest for {}'.format(test_set)
        for key, val in test_set_wers:
            s += '{}\t{}{}\n'.format(key, val, note)
            note=''
        logging.info(s)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
