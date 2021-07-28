#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import logging
import os

from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import k2
import numpy as np
import torch

from k2 import Fsa, SymbolTable

from snowfall.common import average_checkpoint, store_transcripts
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_texts
from snowfall.common import load_checkpoint
from snowfall.common import str2bool
from snowfall.common import write_error_stats
from snowfall.data import LibriSpeechAsrDataModule
from snowfall.decoding.graph import compile_HLG
from snowfall.decoding.lm_rescore import rescore_with_n_best_list
from snowfall.decoding.lm_rescore import rescore_with_whole_lattice
from snowfall.decoding.lm_rescore import compute_am_scores_and_fm_scores
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.text.numericalizer import Numericalizer
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols

def remove_repeated_and_leq(tokens: List[int], blank_id: int = 0):
    '''
    Genrate valid token sequence.
    Result may be used as input of transformer decoder and neural language model.
    Fristly, remove repeated token from a "token alignment" seqs;
    Then remove blank symbols.

    This fuction may be replaced by tokenizing word_seqs with tokenizer
    or composeing word_seqs_fsas with L_inv.fst
    or composing token_seqs with ctc_topo.
    Current method is chosed other than previous three methods because it won't need an extra object, i.e. tokenizer, L.fst or ctc_topo.
    '''
    new_tokens = []
    previous = None
    for token in tokens:
        if token != previous:
            new_tokens.append(token)
            previous = token
    new_tokens = [token for token in new_tokens if token > blank_id]
    return new_tokens

def nbest_am_flm_scrores(lats: k2.Fsa, num_paths: int):
    '''
    Compute am scores with word_seqs
    '''
    # lats has token IDs as labels
    # and word IDs as aux_labels.
    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)
    # word_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.

    word_seqs = k2.index(lats.aux_labels.contiguous(), paths)
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    # lats has token IDs as labels and word IDs as aux_labels.
    unique_word_seqs, _, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=False, need_new2old_indexes=True)

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # used to split final computed tot_scores
    seq_to_path_splits = seq_to_path_shape.row_splits(1)

    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)
    # word_fsas is an FsaVec with axes [path][state][arc]
    word_fsas = k2.linear_fsa(unique_word_seqs)
    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)
    am_scores, lm_scores = compute_am_scores_and_fm_scores(lats, word_fsas_with_epsilon_loops, path_to_seq_map)

    # token_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains token IDs.
    # Note that it also contains 0s and -1s.
    token_seqs = k2.index(lats.labels.contiguous(), paths)
    token_seqs = k2.ragged.remove_axis(token_seqs, 0)
    token_ids, _ = k2.ragged.index(token_seqs, new2old, axis=0)
    token_ids = k2.ragged.to_list(token_ids)
    # Now remove repeated tokens and 0s and -1s.
    token_ids = [remove_repeated_and_leq(tokens) for tokens in token_ids]

    return am_scores, lm_scores, token_ids, new2old, path_to_seq_map, seq_to_path_splits, word_seqs

def nbest_rescoring(model, encoder_memory, memory_mask, lats: k2.Fsa, num_paths: int):
    '''
    N-best rescore with transformer-decoder model.
    The basic idea is to first extra n-best paths from the given lattice.
    Then extract word_seqs and token_seqs for each path.
    Compute the negative log-likehood for each token_seq as 'language model score', called decoder_scores.
    Compute am score for each token_seq.
    Total scores is a weight sum of am_score and decoder_scores.
    The one with the max total score is used as the decoding output.
    '''
    am_scores, fgram_lm_scores, token_ids, new2old, path_to_seq_map, seq_to_path_splits, word_seqs = nbest_am_flm_scrores(lats, num_paths=num_paths)

    # Start to compute lm scores from transformer decoder.

    # an example of path_to_seq_map is:
    # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                device='cuda:0', dtype=torch.int32)
    path_to_seq_map = torch.tensor(path_to_seq_map).to(lats.device)
    seq_to_path_splits = seq_to_path_splits.to('cpu').long()
    num_seqs = len(token_ids)
    # encoder_memory shape: [T, N, C] --> [T, (nbest1 + nbest2 + **), C]
    encoder_memory = encoder_memory.index_select(1, path_to_seq_map)
    # memory_mask shape: [N, T] --> [(nbest1+nbest2), T]
    memory_mask = memory_mask.index_select(0, path_to_seq_map)

    # nll: negative log-likelihood
    nll = model.decoder_nll(encoder_memory, memory_mask, token_ids=token_ids)
    assert nll.shape[0] == num_seqs
    decoder_scores = - nll.sum(dim=1)

    flm_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
    decoder_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]

    ans = dict()
    word_seqs = k2.ragged.to_list(k2.ragged.remove_axis(word_seqs,0))
    for flm_scale in flm_scale_list:
        for decoder_scale in decoder_scale_list:
            key = f'lm_scale_{flm_scale}_decoder_scale_{decoder_scale}'
            batch_tot_scores = am_scores + flm_scale * fgram_lm_scores + decoder_scale * decoder_scores
            batch_tot_scores = torch.tensor_split(batch_tot_scores, seq_to_path_splits[1:])
            ans[key] = []
            processed_seqs = 0
            for tot_scores in batch_tot_scores:
                if tot_scores.nelement() == 0:
                    # the last element by torch.tensor_split may be empty
                    # e.g.
                    # torch.tensor_split(torch.tensor([1,2,3,4]), torch.tensor([2,4]))
                    # (tensor([1, 2]), tensor([3, 4]), tensor([], dtype=torch.int64))

                    break
                best_seq_idx = new2old[processed_seqs + torch.argmax(tot_scores)]
                best_word_seq = word_seqs[best_seq_idx]
                processed_seqs += tot_scores.nelement()
                ans[key].append(best_word_seq)
            assert len(ans[key]) == seq_to_path_splits.nelement() - 1

    return ans

def decode_one_batch(batch: Dict[str, Any],
                     model: AcousticModel,
                     HLG: k2.Fsa,
                     output_beam_size: float,
                     num_paths: int,
                     use_whole_lattice: bool,
                     nbest_rescore_with_decoder: bool = True,
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
        It specifies the size of `n` in n-best list decoding with transforer decoder model.
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
    batch_size = feature.shape[0]

    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]

    supervisions = batch['supervisions']

    nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
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
    # cuts has been sorted in lhotse dataset
    # https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L109
    assert torch.all(torch.argsort(indices) == indices)

    supervision_segments = supervision_segments[indices]

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

    lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

    # fgram means four-gram
    fgram_rescored_lattices = rescore_with_whole_lattice(lattices, G,
                                                 lm_scale_list=None,
                                                 need_rescored_lats=True)
    ans = nbest_rescoring(model, encoder_memory, memory_mask, fgram_rescored_lattices, num_paths)
    return ans


@torch.no_grad()
def decode(dataloader: torch.utils.data.DataLoader,
           model: AcousticModel,
           HLG: Fsa,
           symbols: SymbolTable,
           num_paths: int,
           G: k2.Fsa,
           use_whole_lattice: bool,
           output_beam_size: float):
    del HLG.lm_scores
    HLG.lm_scores = HLG.scores.clone()
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
            for hyp, text in zip(hyps, texts):
                hyp_words = [symbols.get(x) for x in hyp]
                ref_words = text.split(' ')
                this_batch.append((ref_words, hyp_words))

            results[lm_scale].extend(this_batch)

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))

        num_cuts += len(texts)

    return results


def get_parser():
    parser = argparse.ArgumentParser(
        description="ASR Decoding with bpe model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["conformer"],
        help="Model type.")
    parser.add_argument(
        '--epoch',
        type=int,
        default=35,
        help="Decoding epoch.")
    parser.add_argument(
        '--avg',
        type=int,
        default=10,
        help="Number of checkpionts to average. Automatically select "
             "consecutive checkpoints before checkpoint specified by'--epoch'. ")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.7,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=8,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=512,
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
        '--num-paths-for-decoder-rescore',
        type=int,
        default=500,
        help='Number of paths for rescoring using n-best list with transformer decoder model.')

    parser.add_argument(
        '--is-espnet-structure',
        type=str2bool,
        default=True,
        help='When enabled, the conformer will have the ' \
             'same structure like espnet')

    parser.add_argument(
        '--vgg-frontend',
        type=str2bool,
        default=False,
        help='When enabled, it uses vgg style network for subsampling')

    parser.add_argument(
        '--espnet-identical-model',
        type=str2bool,
        default=False,
        help='When enabled, train an identical model to the espnet SOTA released model'
        "url: https://zenodo.org/record/4604066#.YNAAHmgzZPY")

    parser.add_argument(
        '--generate-release-model',
        type=str2bool,
        default=False,
        help='When enabled, save the averaged model to release')

    parser.add_argument(
        '--decode_with_released_model',
        type=str2bool,
        default=False,
        help='When enabled, decode and evaluate with the released model')

    parser.add_argument(
        '--lr-factor',
        type=float,
        default=10.0,
        help='Learning rate factor for Noam optimizer.'
    )


    return parser


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    avg = args.avg
    attention_dim = args.attention_dim
    nhead=args.nhead
    att_rate = args.att_rate
    model_type = args.model_type
    epoch = args.epoch
    use_lm_rescoring = args.use_lm_rescoring
    use_whole_lattice = True

    output_beam_size = args.output_beam_size

    # load L, G, symbol_table
    logging.debug("About to load phone and word symbols")
    lang_dir = Path('./data/lang_bpe2/')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)

    logging.debug("About to load model")
    # Note: Use "export CUDA_VISIBLE_DEVICES=N" to setup device id to N
    # device = torch.device('cuda', 1)
    device = torch.device('cuda')


    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    num_classes = len(phone_ids) + 1
    assert num_classes == 5000, print(num_classes)
    if model_type == "conformer":
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes = num_classes,
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=args.vgg_frontend,
            is_espnet_structure=args.is_espnet_structure,
            mmi_loss=False,
            use_feat_batchnorm=True)

        if args.espnet_identical_model:
            # + 160 for feat_batch_norm, used as feature mean and variance normalization
            assert sum([p.numel() for p in model.parameters()]) == 116146960 + 160
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not verified")

    exp_dir = Path(f'exp-duration-200-feat_batchnorm-bpe-lrfactor{args.lr_factor}-{model_type}-{attention_dim}-{nhead}-noam/')

    if args.decode_with_released_model is True:
        released_model_path = exp_dir / f'model-epoch-{epoch}-avg-{avg}.pt'
        model.load_state_dict(torch.load(released_model_path))
    else:
        if avg == 1:
            checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
            load_checkpoint(checkpoint, model)
        else:
            checkpoints = [os.path.join(exp_dir, 'epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                           range(epoch - avg, epoch)]
            average_checkpoint(checkpoints, model)
        if args.generate_release_model:
            released_model_path = exp_dir / f'model-epoch-{epoch}-avg-{avg}.pt'
            torch.save(model.state_dict(), released_model_path)

    model.to(device)
    model.eval()
    token_ids_with_blank = [i for i in range(num_classes)]

    ctc_path = lang_dir / 'ctc_topo.pt'

    if not os.path.exists(ctc_path):
        logging.info("Generating ctc topo...")
        ctc_topo = k2.arc_sort(build_ctc_topo(token_ids_with_blank))
        torch.save(ctc_topo.as_dict(), ctc_path)

    else:
        logging.info("Loading pre-compiled ctc topo fst")
        d_ctc_topo = torch.load(ctc_path)
        ctc_topo = k2.Fsa.from_dict(d_ctc_topo)

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
        logging.debug('Currently 4-gram lattice rescore is required.')
        sys.exit()

    logging.debug("convert HLG to device")
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    if not hasattr(HLG, 'lm_scores'):
        HLG.lm_scores = HLG.scores.clone()

    librispeech = LibriSpeechAsrDataModule(args)
    test_sets = ['test-clean', 'test-other']
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        logging.info(f'* DECODING: {test_set}')

        test_set_wers = dict()
        results_dict = decode(dataloader=test_dl,
                              model=model,
                              HLG=HLG,
                              symbols=symbol_table,
                              num_paths=args.num_paths_for_decoder_rescore,
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
