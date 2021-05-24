#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import logging
import os
import random
import re
import sys

from pathlib import Path
from typing import Union

import k2
import numpy as np
import torch

from kaldialign import edit_distance
from lhotse import load_manifest
from lhotse.dataset import K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.dataset.input_strategies import AudioSamples

from espnet_utils.asr import ESPnetASRModel
from espnet_utils.nnlm_evaluator import EspnetNNLMEvaluator
from snowfall.common import store_transcripts
from snowfall.common import write_error_stats
from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from snowfall.training.ctc_graph import build_ctc_topo


def decode(dataloader: torch.utils.data.DataLoader,
           model: None,
           device: Union[str, torch.device],
           ctc_topo: None,
           G=None,
           evaluator=None,
           numericalizer=None,
           num_paths=-1):
    tot_num_cuts = len(dataloader.dataset.cuts)
    num_cuts = 0
    results = []
    for batch_idx, batch in enumerate(dataloader):
        assert isinstance(batch, dict), type(batch)
        speech = batch['inputs'].squeeze()
        lengths = batch['supervisions']['num_samples']
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0)
        speech = speech.to(torch.device(device))
        lengths = lengths.to(torch.device(device))

        nnet_output = model.encode(speech=speech, speech_lengths=lengths)
        nnet_output = nnet_output.detach()

        blank_bias = -1.0
        nnet_output[:, :, 0] += blank_bias

        supervision_segments = torch.tensor([[0, 0, nnet_output.shape[1]]],
                                            dtype=torch.int32)

        with torch.no_grad():
            dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

            output_beam_size = 8
            lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0,
                                                 output_beam_size, 30, 10000)

        use_whole_lattice = False
        best_paths = decode_with_lm_rescoring(
            lattices,
            G,
            evaluator,
            num_paths=num_paths,
            use_whole_lattice=use_whole_lattice)

        token_int = list(
            filter(
                lambda x: x not in
                [-1, 0, numericalizer.sos_idx, numericalizer.eos_idx],
                best_paths.aux_labels.cpu().numpy()))

        token = numericalizer.ids2tokens(token_int)

        text = numericalizer.tokens2text(token)

        ref = batch['supervisions']['text']
        for i in range(len(ref)):
            hyp_words = text.split(' ')
            ref_words = ref[i].split(' ')
            results.append((ref_words, hyp_words))
        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))
        num_cuts += 1
    return results


def get_parser():
    parser = argparse.ArgumentParser(
        description="ASR Decoding with model from espnet model zoo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--seed", type=int, default=2021, help="Random seed")

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument('--lm_train_config', type=str, required=True)
    group.add_argument('--lm_model_file', type=str, required=True)
    group.add_argument('--num_paths', type=int, required=True)

    return parser


def main():
    parser = get_parser()
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    asr_train_config = args.asr_train_config
    asr_model_file = args.asr_model_file
    seed = args.seed

    device = "cuda"

    # 1. Set random-seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    asr_model, numericalizer = ESPnetASRModel.build_model(
        asr_train_config, asr_model_file, device)

    asr_model.eval()

    phone_ids_with_blank = [i for i in range(len(numericalizer.token_list))]

    exp_dir = Path('exp/')
    ctc_path = exp_dir / 'ctc_topo.pt'

    if not os.path.exists(ctc_path):
        logging.info("Generating ctc topo...")
        ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
        torch.save(ctc_topo.as_dict(), ctc_path)

    else:
        logging.info("Loading pre-compiled ctc topo fst")
        d_ctc_topo = torch.load(ctc_path)
        ctc_topo = k2.Fsa.from_dict(d_ctc_topo)
    ctc_topo = ctc_topo.to(device)

    evaluator = EspnetNNLMEvaluator.build_model(args.lm_train_config,
                                                args.lm_model_file,
                                                device=device,
                                                input_type='auxlabel',
                                                numericalizer=numericalizer)
    evaluator.lm.eval()
    feature_dir = Path('exp/data')

    test_sets = ['test-clean', 'test-other']
    for test_set in test_sets:
        cuts_test = load_manifest(feature_dir / f'cuts_{test_set}.json.gz')
        sampler = SingleCutSampler(cuts_test, max_cuts=1)

        test = K2SpeechRecognitionDataset(cuts_test,
                                          input_strategy=AudioSamples())
        test_dl = torch.utils.data.DataLoader(test,
                                              batch_size=None,
                                              sampler=sampler)
        results = decode(dataloader=test_dl,
                         model=asr_model,
                         device=device,
                         ctc_topo=ctc_topo,
                         evaluator=evaluator,
                         numericalizer=numericalizer,
                         num_paths=args.num_paths)

        recog_path = exp_dir / f'recogs-{test_set}.txt'
        store_transcripts(path=recog_path, texts=results)
        logging.info(f'The transcripts are stored in {recog_path}')

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = exp_dir / f'errs-{test_set}.txt'
        with open(errs_filename, 'w') as f:
            write_error_stats(f, test_set, results)
        logging.info('Wrote detailed error stats to {}'.format(errs_filename))

        dists = [edit_distance(r, h) for r, h in results]
        errors = {
            key: sum(dist[key] for dist in dists)
            for key in ['sub', 'ins', 'del', 'total']
        }
        total_words = sum(len(ref) for ref, _ in results)
        # Print Kaldi-like message:
        # %WER 2.62 [ 1380 / 52576, 176 ins, 106 del, 1098 sub ]
        logging.info(
            f'[{test_set}] %WER {errors["total"] / total_words:.2%} '
            f'[{errors["total"]} / {total_words}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
        )


if __name__ == "__main__":
    main()
