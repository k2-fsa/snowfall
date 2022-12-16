#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu, Fangjun Kuang)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import k2
import logging
import numpy as np
import os
import sys
import torch
from k2 import Fsa, SymbolTable
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from snowfall.common2 import average_checkpoint, average_checkpoint_2nd, store_transcripts
from snowfall.common2 import find_first_disambig_symbol
from snowfall.common2 import get_texts
from snowfall.common2 import write_error_stats
from snowfall.common2 import load_checkpoint
from snowfall.common2 import setup_logger
from snowfall.common2 import str2bool
from snowfall.data import LibriSpeechAsrDataModule
from snowfall.decoding.graph import compile_HLG
from snowfall.decoding.lm_rescore2 import decode_with_lm_rescoring
from snowfall.models import AcousticModel
from snowfall.models.transformer import Transformer
from snowfall.models.conformer import Conformer
from snowfall.models.second_pass_model import SecondPassModel
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph2 import create_bigram_phone_lm
from snowfall.training.mmi_graph2 import get_phone_symbols


# TODO(fangjun): Replace it with
# https://github.com/k2-fsa/snowfall/issues/232
@torch.no_grad()
def second_pass_decoding(second_pass: AcousticModel,
                         lats: k2.Fsa,
                         supervision_segments: torch.Tensor,
                         encoder_memory: torch.Tensor,
                         num_paths: int = 10):
    '''
    The fundamental idea is to get an n-best list from the first pass
    decoding lattice and use the second pass model to do rescoring.
    The path with the highest score is used as the final decoding output.

    Args:
      lats:
        It's the decoding lattice from the first pass.
    '''
    device = lats.device
    assert len(lats.shape) == 3

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # phone_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains phone IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    phone_seqs = k2.index(lats.labels.clone(), paths)

    num_seqs = phone_seqs.dim0()

    indexes = torch.arange(start=0,
                           end=(num_seqs - 1) * num_paths + 1,
                           step=num_paths,
                           dtype=torch.int32,
                           device=device)
    # From indexes, we can select
    #
    # (1) path 0 of seq0, seq1, seq2, ...
    # (2) path 1 of seq0, seq1, seq2, ...
    # (3) path 2 of seq0, seq1, seq2, ...

    paths_offset = phone_seqs.row_splits(1)
    phone_seqs = k2.ragged.remove_axis(phone_seqs, 0)
    # Note, from now on phone_seqs has only two axes

    phone_seqs = k2.ragged.remove_values_leq(phone_seqs, -1)
    log_probs = torch.empty(num_paths, num_seqs)
    for i in range(num_paths):
        # path is a k2.RaggedInt with axes [path][phones], excluding -1
        p, _ = k2.ragged.index(phone_seqs, indexes + i, axis=0)
        phone_fsa = k2.linear_fsa(p)
        nnet_output_2nd = second_pass(encoder_memory, phone_fsa, supervision_segments)

        nnet_output_2nd = nnet_output_2nd.cpu()
        row_splits1 = p.row_splits(1).cpu()
        value = p.values().cpu()

        for k in range(num_seqs):
            nnet_output_this_seq = nnet_output_2nd[k]
            start = row_splits1[k]
            end = row_splits1[k+1]
            num_frames = end - start

            this_value = value[start:end].tolist()
            log_p = 0
            for idx, v in enumerate(this_value):
                log_p += nnet_output_this_seq[idx, v]
            log_p /= num_frames
            log_probs[i, k] = log_p

    # Now get the best score of the path within n-best list
    # of each seq
    best_indexes = torch.argmax(log_probs, dim=0).to(torch.int32)
    best_paths_indexes = best_indexes.to(device) + paths_offset[:-1]

    # best_paths has three axes [seq][path][arc_pos]
    # each seq has only one path
    best_paths, _ = k2.ragged.index(paths, best_paths_indexes, axis=1)

    best_phone_seqs = k2.index(lats.labels.clone(), best_paths)

    best_phone_seqs = k2.ragged.remove_values_leq(best_phone_seqs, -1)
    best_phone_seqs = k2.ragged.remove_axis(best_phone_seqs, 0)

    ans = k2.linear_fsa(best_phone_seqs)
    ans.aux_labels = k2.index(lats.aux_labels, best_paths.values())

    return ans



@torch.no_grad()
def decode(dataloader: torch.utils.data.DataLoader, model: AcousticModel,
           second_pass: Optional[AcousticModel],
           device: Union[str, torch.device], HLG: Fsa, symbols: SymbolTable,
           num_paths: int, G: k2.Fsa, use_whole_lattice: bool, output_beam_size: float):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = []  # a list of pair (ref_words, hyp_words)
    for batch_idx, batch in enumerate(dataloader):
        feature = batch['inputs']
        supervisions = batch['supervisions']
        supervision_segments = torch.stack(
            (supervisions['sequence_idx'],
             (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
             (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
        supervision_segments = torch.clamp(supervision_segments, min=0)
        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        texts = supervisions['text']
        assert feature.ndim == 3

        feature = feature.to(device)
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        with torch.no_grad():
            nnet_output, encoder_memory, _ = model(feature, supervisions)
        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2,
                                          1)  # now nnet_output is [N, T, C]

        #  blank_bias = -3.0
        #  nnet_output[:, :, 0] += blank_bias

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
        # assert HLG.is_cuda()
        assert HLG.device == nnet_output.device, \
            f"Check failed: HLG.device ({HLG.device}) == nnet_output.device ({nnet_output.device})"
        # TODO(haowen): with a small `beam`, we may get empty `target_graph`,
        # thus `tot_scores` will be `inf`. Definitely we need to handle this later.
        beam_size = output_beam_size
        while True:
            try:
                if beam_size < 2:
                    logging.error(f'beam size {beam_size} is too small')
                    sys.exit(1)
                lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, beam_size, 30,
                                                     10000)
                if second_pass is not None:
                    best_paths = second_pass_decoding(
                        second_pass=second_pass,
                        lats=lattices,
                        supervision_segments=supervision_segments,
                        encoder_memory=encoder_memory,
                        num_paths=num_paths)
                else:
                    if G is None:
                        best_paths = k2.shortest_path(lattices, use_double_scores=True)
                    else:
                        best_paths = decode_with_lm_rescoring(
                            lattices,
                            G,
                            num_paths=num_paths,
                            use_whole_lattice=use_whole_lattice)
                break
            except RuntimeError as e:
                logging.info(f'Caught exception:\n{e}\n')
                new_beam_size = beam_size * 0.95
                logging.info(f'Change beam_size from {beam_size} to {new_beam_size}')
                beam_size = new_beam_size

        #  if second_pass is not None:
        #      # Now for the second pass model
        #      best_paths = k2.shortest_path(lattices, use_double_scores=True)
        #
        #      nnet_output_2nd = second_pass(encoder_memory, best_paths, supervision_segments)
        #      # nnet_output_2nd is [N, T, C]
        #
        #      assert nnet_output_2nd.shape[0] == supervision_segments.shape[0]
        #
        #      supervision_segments_2nd = supervision_segments.clone()
        #
        #      # [0, 1, 2, 3, ...]
        #      supervision_segments_2nd[:, 0] = torch.arange(supervision_segments_2nd.shape[0], dtype=torch.int32)
        #
        #      # start offset
        #      supervision_segments_2nd[:, 1] = 0
        #
        #      # duration of supervision_segments_2nd is kept unchanged
        #
        #      dense_fsa_vec_2nd = k2.DenseFsaVec(nnet_output_2nd, supervision_segments_2nd)
        #
        #      lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec_2nd, 20.0, output_beam_size, 30,
        #                                           10000)

        #  if G is None:
        #      best_paths = k2.shortest_path(lattices, use_double_scores=True)
        #  else:
        #      best_paths = decode_with_lm_rescoring(
        #          lattices,
        #          G,
        #          num_paths=num_paths,
        #          use_whole_lattice=use_whole_lattice)

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
    table[0, -1] = 0  # the start state has no arcs to the final state
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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer"],
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
        help="Number of checkpionts to average. Automaticly select "
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
        '--use-second-pass',
        type=str2bool,
        default=True,
        help='When enabled, it uses the second pass model')
    parser.add_argument(
        '--num-paths',
        type=int,
        default=-1,
        help='Number of paths for rescoring using n-best list.' \
             'If it is negative, then rescore with the whole lattice.'\
             'CAUTION: You have to reduce max_duration in case of CUDA OOM'
             )
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
    use_second_pass = args.use_second_pass

    exp_dir = Path('exp-' + model_type + '-noam-mmi-att-musan-sa-new')
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
            num_decoder_layers=num_decoder_layers)
    else:
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers)

    model.P_scores = torch.nn.Parameter(P.scores.clone(), requires_grad=False)

    if use_second_pass:
        second_pass = SecondPassModel(max_phone_id=max(phone_ids)).to(device)

    if avg == 1:
        checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
        load_checkpoint(checkpoint, model)

        if use_second_pass:
            checkpoint_2nd = os.path.join(exp_dir, '2nd-epoch-' + str(epoch - 1) + '.pt')
            logging.info(f'loading {checkpoint_2nd}')
            second_pass.load_state_dict(torch.load(checkpoint_2nd, map_location='cpu'))
    else:
        checkpoints = [os.path.join(exp_dir, 'epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                       range(epoch - avg, epoch)]
        average_checkpoint(checkpoints, model)

        if use_second_pass:
            checkpoints_2nd = [os.path.join(exp_dir, '2nd-epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                           range(epoch - avg, epoch)]
            average_checkpoint_2nd(checkpoints_2nd, second_pass)

    model.to(device)
    model.eval()

    if use_second_pass:
        second_pass.to(device)
        second_pass.eval()
    else:
        second_pass = None
        logging.info('Not using the second pass model')

    assert P.requires_grad is False
    P.scores = model.P_scores.cpu()
    print_transition_probabilities(P, phone_symbol_table, phone_ids, filename='model_P_scores.txt')

    P.set_scores_stochastic_(model.P_scores)
    print_transition_probabilities(P, phone_symbol_table, phone_ids, filename='P_scores.txt')

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
        G.lm_scores = G.scores.clone()
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
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        logging.info(f'* DECODING: {test_set}')

        results = decode(dataloader=test_dl,
                         model=model,
                         second_pass=second_pass,
                         device=device,
                         HLG=HLG,
                         symbols=symbol_table,
                         num_paths=num_paths,
                         G=G,
                         use_whole_lattice=use_whole_lattice,
                         output_beam_size=output_beam_size)

        recog_path = exp_dir / f'recogs-{test_set}.txt'
        store_transcripts(path=recog_path, texts=results)
        logging.info(f'The transcripts are stored in {recog_path}')

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = exp_dir / f'errs-{test_set}.txt'
        with open(errs_filename, 'w') as f:
            write_error_stats(f, test_set, results)
        logging.info('Wrote detailed error stats to {}'.format(errs_filename))


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
