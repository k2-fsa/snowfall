#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import logging
import os
from pathlib import Path
from typing import Union

import k2
import torch

from snowfall.common import average_checkpoint, store_transcripts
from snowfall.common import get_texts
from snowfall.common import load_checkpoint
from snowfall.common import str2bool
from snowfall.common import write_error_stats
from snowfall.data import LibriSpeechAsrDataModule
from snowfall.models.conformer import Conformer
from snowfall.text.numericalizer import Numericalizer
from snowfall.training.ctc_graph import build_ctc_topo


def decode(dataloader: torch.utils.data.DataLoader,
           model: None,
           device: Union[str, torch.device],
           ctc_topo: None,
           numericalizer=None,
           num_paths=-1,
           output_beam_size: float = 8):
    num_batches = None
    try:
        num_batches = len(dataloader)
    except AttributeError:
        pass
    num_cuts = 0
    results = []
    for batch_idx, batch in enumerate(dataloader):
        assert isinstance(batch, dict), type(batch)
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
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        nnet_output = nnet_output.permute(0, 2, 1)

        # TODO(Liyong Guo): Tune this bias
        # blank_bias = 0.0
        # nnet_output[:, :, 0] += blank_bias

        with torch.no_grad():
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

            lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0,
                                                 output_beam_size, 30, 10000)

        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        hyps = get_texts(best_paths, indices)
        assert len(hyps) == len(texts)

        for i in range(len(texts)):
            pieces = [numericalizer.tokens_list[token_id] for token_id in hyps[i]]
            hyp_words = numericalizer.tokenizer.DecodePieces(pieces).split(' ')
            ref_words = texts[i].split(' ')
            results.append((ref_words, hyp_words))

        if batch_idx % 10 == 0:
            batch_str = f"{batch_idx}" if num_batches is None else f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, number of cuts processed until now is {num_cuts}")
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
        default=False,
        help='When enabled, it uses vgg style network for subsampling')

    parser.add_argument(
        '--espnet-identical-model',
        type=str2bool,
        default=False,
        help='When enabled, train an identical model to the espnet SOTA released model'
        "url: https://zenodo.org/record/4604066#.YNAAHmgzZPY")
    parser.add_argument(
        '--epoch',
        type=int,
        default=29,
        help="Decoding epoch.")
    parser.add_argument(
        '--avg',
        type=int,
        default=5,
        help="Number of checkpionts to average. Automaticly select "
             "consecutive checkpoints before checkpoint specified by'--epoch'. ")

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
        '--att-rate',
        type=float,
        default=0.7,
        help="Attention loss rate.")

    parser.add_argument(
        '--output-beam-size',
        type=float,
        default=8,
        help='Output beam size. Used in k2.intersect_dense_pruned.')

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

    # Note: Use "export CUDA_VISIBLE_DEVICES=N" to setup device id to N
    # device = torch.device('cuda', 1)
    device = torch.device('cuda')

    lang_dir = Path('data/en_token_list/bpe_unigram5000/')
    bpe_model_path = lang_dir / 'bpe.model'
    tokens_file = lang_dir / 'tokens.txt'
    numericalizer = Numericalizer.build_numericalizer(bpe_model_path, tokens_file)

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    num_classes = len(numericalizer.tokens_list)
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
            mmi_loss=False)

        if args.espnet_identical_model:
            assert sum([p.numel() for p in model.parameters()]) == 116146960
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not verified")

    exp_dir = Path(f'exp-bpe-{model_type}-{attention_dim}-{nhead}-noam/')
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
    ctc_topo = ctc_topo.to(device)

    feature_dir = Path('exp/data')

    librispeech = LibriSpeechAsrDataModule(args)
    test_sets = ['test-clean', 'test-other']
    for test_set, test_dl in zip(test_sets, librispeech.test_dataloaders()):
        results = decode(dataloader=test_dl,
                         model=model,
                         device=device,
                         ctc_topo=ctc_topo,
                         numericalizer=numericalizer,
                         num_paths=args.num_paths,
                         output_beam_size=args.output_beam_size)

        recog_path = exp_dir / f'recogs-{test_set}.txt'
        store_transcripts(path=recog_path, texts=results)
        logging.info(f'The transcripts are stored in {recog_path}')

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = exp_dir / f'errs-{test_set}.txt'
        with open(errs_filename, 'w') as f:
            write_error_stats(f, test_set, results)
        logging.info('Wrote detailed error stats to {}'.format(errs_filename))

if __name__ == "__main__":
    main()
