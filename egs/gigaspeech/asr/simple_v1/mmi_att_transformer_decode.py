#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
#                2020  Johns Hopkins University (author: Piotr Żelasko)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

# Usage of this script:
'''

# Without LM rescoring

## Use n-best decoding
./mmi_att_transformer_decode.py \
  --use-lm-rescoring=0 \
  --num-paths=100 \
  --max-duration=300

## Use 1-best decoding
./mmi_att_transformer_decode.py \
  --use-lm-rescoring=0 \
  --num-paths=1 \
  --max-duration=300

# With LM rescoring

## Use whole lattice
./mmi_att_transformer_decode.py \
  --use-lm-rescoring=1 \
  --num-paths=-1 \
  --max-duration=300

## Use n-best list
./mmi_att_transformer_decode.py \
  --use-lm-rescoring=1 \
  --num-paths=100 \
  --max-duration=300
'''

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import k2
import torch
from k2 import Fsa, SymbolTable

from snowfall.common import average_checkpoint, store_transcripts, store_transcripts_for_sclite
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_texts
from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.common import str2bool
from snowfall.common import write_error_stats
from snowfall.data.gigaspeech import GigaSpeechAsrDataModule
from snowfall.decoding.graph import compile_HLG
from snowfall.decoding.lm_rescore import rescore_with_n_best_list
from snowfall.decoding.lm_rescore import rescore_with_whole_lattice
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.contextnet import ContextNet
from snowfall.models.transformer import Transformer
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols


def nbest_decoding(lats: k2.Fsa, num_paths: int):
    '''
    (Ideas of this function are from Dan)

    It implements something like CTC prefix beam search using n-best lists

    The basic idea is to first extra n-best paths from the given lattice,
    build a word seqs from these paths, and compute the total scores
    of these sequences in the log-semiring. The one with the max score
    is used as the decoding output.
    '''

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # word_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.

    word_seqs = k2.index(lats.aux_labels, paths)
    # Note: the above operation supports also the case when
    # lats.aux_labels is a ragged tensor. In that case,
    # `remove_axis=True` is used inside the pybind11 binding code,
    # so the resulting `word_seqs` still has 3 axes, like `paths`.
    # The 3 axes are [seq][path][word]

    # Remove epsilons and -1 from word_seqs
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    # Remove repeated sequences to avoid redundant computation later.
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.num_elements()
    unique_word_seqs, _, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=False, need_new2old_indexes=True)
    # Note: unique_word_seqs still has the same axes as word_seqs

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seqs has only two axes [path][word]
    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)

    # word_fsas is an FsaVec with axes [path][state][arc]
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    # lats has phone IDs as labels and word IDs as aux_labels.
    # inv_lats has word IDs as labels and phone IDs as aux_labels
    inv_lats = k2.invert(lats)
    inv_lats = k2.arc_sort(inv_lats) # no-op if inv_lats is already arc-sorted

    path_lats = k2.intersect_device(inv_lats,
                                    word_fsas_with_epsilon_loops,
                                    b_to_a_map=path_to_seq_map,
                                    sorted_match_a=True)
    # path_lats has word IDs as labels and phone IDs as aux_labels

    path_lats = k2.top_sort(k2.connect(path_lats.to('cpu')).to(lats.device))

    tot_scores = path_lats.get_tot_scores(True, True)
    # RaggedFloat currently supports float32 only.
    # We may bind Ragged<double> as RaggedDouble if needed.
    ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape,
                                       tot_scores.to(torch.float32))

    argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

    # Since we invoked `k2.ragged.unique_sequences`, which reorders
    # the index from `paths`, we use `new2old`
    # here to convert argmax_indexes to the indexes into `paths`.
    #
    # Use k2.index here since argmax_indexes' dtype is torch.int32
    best_path_indexes = k2.index(new2old, argmax_indexes)

    paths_2axes = k2.ragged.remove_axis(paths, 0)

    # best_paths is a k2.RaggedInt with 2 axes [path][arc_pos]
    best_paths = k2.index(paths_2axes, best_path_indexes)

    # labels is a k2.RaggedInt with 2 axes [path][phone_id]
    # Note that it contains -1s.
    labels = k2.index(lats.labels.contiguous(), best_paths)

    labels = k2.ragged.remove_values_eq(labels, -1)

    # lats.aux_labels is a k2.RaggedInt tensor with 2 axes, so
    # aux_labels is also a k2.RaggedInt with 2 axes
    aux_labels = k2.index(lats.aux_labels, best_paths.values())

    best_path_fsas = k2.linear_fsa(labels)
    best_path_fsas.aux_labels = aux_labels

    return best_path_fsas


def decode_one_batch(batch: Dict[str, Any],
                     model: AcousticModel,
                     HLG: k2.Fsa,
                     output_beam_size: float,
                     num_paths: int,
                     use_whole_lattice: bool,
                     G: Optional[k2.Fsa] = None)->Dict[str, List[List[int]]]:
    '''
    Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if no rescoring is used, the key is the string `no_rescore`.
               If LM rescoring is used, the key is the string `lm_scale_xxx`,
               where `xxx` is the value of `lm_scale`. An example key is
               `lm_scale_0.7`
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.

    Args:
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      model:
        The neural network model.
      HLG:
        The decoding graph.
      output_beam_size:
        Size of the beam for pruning.
      use_whole_lattice:
        If True, `G` must not be None and it will use whole lattice for
        LM rescoring.
        If False and if `G` is not None, then `num_paths` must be positive
        and it will use n-best list for LM rescoring.
      num_paths:
        It specifies the size of `n` in n-best list decoding.
      G:
        The LM. If it is None, no rescoring is used.
        Otherwise, LM rescoring is used.
        It supports two types of LM rescoring: n-best list rescoring
        and whole lattice rescoring.
        `use_whole_lattice` specifies which type to use.

    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    '''
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
        if num_paths > 1:
            best_paths = nbest_decoding(lattices, num_paths)
            key=f'no_rescore-{num_paths}'
        else:
            key = 'no_rescore'
            best_paths = k2.shortest_path(lattices, use_double_scores=True)
        hyps = get_texts(best_paths, indices)
        return {key: hyps}

    lm_scale_list = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    lm_scale_list += [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    if use_whole_lattice:
        best_paths_dict = rescore_with_whole_lattice(lattices, G,
                                                     lm_scale_list)
    else:
        best_paths_dict = rescore_with_n_best_list(lattices, G, num_paths,
                                                   lm_scale_list)
    # best_paths_dict is a dict
    #  - key: lm_scale_xxx, where xxx is the value of lm_scale. An example
    #         key is lm_scale_1.2
    #  - value: it is the best path obtained using the corresponding lm scale
    #           from the dict key.

    ans = dict()
    for lm_scale_str, best_paths in best_paths_dict.items():
        hyps = get_texts(best_paths, indices)
        ans[lm_scale_str] = hyps
    return ans


@torch.no_grad()
def decode(dataloader: torch.utils.data.DataLoader, model: AcousticModel,
           HLG: Fsa, symbols: SymbolTable,
           num_paths: int, G: k2.Fsa, use_whole_lattice: bool, output_beam_size: float):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = defaultdict(list)
    # results is a dict whose keys and values are:
    #  - key: It indicates the lm_scale, e.g., lm_scale_1.2.
    #         If no rescoring is used, the key is the literal string: no_rescore
    #
    #  - value: It is a list of tuples (ref_words, hyp_words)

    for batch_idx, batch in enumerate(dataloader):
        texts = batch['supervisions']['text']

        hyps_dict = decode_one_batch(batch=batch,
                                     model=model,
                                     HLG=HLG,
                                     output_beam_size=output_beam_size,
                                     num_paths=num_paths,
                                     use_whole_lattice=use_whole_lattice,
                                     G=G)

        for lm_scale, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)

            for i in range(len(texts)):
                hyp_words = [symbols.get(x) for x in hyps[i]]
                ref_words = texts[i].split(' ')
                this_batch.append((ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))

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
    GigaSpeechAsrDataModule.add_arguments(parser)
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

    suffix = ''
    if args.context_window is not None and args.context_window > 0:
        suffix = f'ac{args.context_window}'
    giga_subset = f'giga{args.subset}'
    exp_dir = Path(f'exp-{model_type}-mmi-att-sa-vgg-normlayer-{giga_subset}-{suffix}')

    setup_logger('{}/log/log-decode'.format(exp_dir), log_level='debug')

    logging.info(f'output_beam_size: {output_beam_size}')

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')

    phone_ids = get_phone_symbols(phone_symbol_table)

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
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        logging.debug('Decoding without LM rescoring')
        G = None
        if num_paths > 1:
            logging.debug(f'Use n-best list decoding, n is {num_paths}')
        else:
            logging.debug('Use 1-best decoding')

    logging.debug("convert HLG to device")
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    if not hasattr(HLG, 'lm_scores'):
        HLG.lm_scores = HLG.scores.clone()

    # load dataset
    gigaspeech = GigaSpeechAsrDataModule(args)
    test_sets = ['DEV', 'TEST']
    for test_set, test_dl in zip(test_sets, [gigaspeech.valid_dataloaders(), gigaspeech.test_dataloaders()]):
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

            ref_path = exp_dir / f'ref-{test_set}.trn'
            hyp_path = exp_dir / f'hyp-{test_set}.trn'
            store_transcripts_for_sclite(
                ref_path=ref_path,
                hyp_path=hyp_path,
                texts=results
            )
            logging.info(f'The sclite-format transcripts are stored in {ref_path} and {hyp_path}')
            cmd = f'python3 GigaSpeech/utils/gigaspeech_scoring.py {ref_path} {hyp_path} {exp_dir / "tmp_sclite"}'
            logging.info(cmd)
            try:
                subprocess.run(cmd, check=True, shell=True)
            except subprocess.CalledProcessError:
                logging.error('Skipping sclite scoring as it failed to run: Is "sclite" registered in your $PATH?"')

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
