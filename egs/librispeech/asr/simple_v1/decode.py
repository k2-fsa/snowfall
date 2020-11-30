#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
# Apache 2.0

import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import math

import editdistance
import numpy as np
import torch
import torchaudio
import torchaudio.models
import k2

from lhotse import CutSet, Fbank, LilcomFilesWriter, WavAugmenter
from lhotse.dataset import SpeechRecognitionDataset
from lhotse.dataset.speech_recognition import K2DataLoader, K2SpeechRecognitionDataset, \
    K2SpeechRecognitionIterableDataset, concat_cuts
from lhotse.recipes.librispeech import download_and_untar, prepare_librispeech, dataset_parts_full

from common import load_checkpoint
from common import setup_logger
from model import Model


def decode(dataloader, model, subsampling, device, LG, symbols):
    results = []  # a list of pair (ref_words, hyp_words)
    for batch_idx, batch in enumerate(dataloader):
        feature = batch['features']
        supervisions = batch['supervisions']
        supervision_segments = torch.stack(
            (supervisions['sequence_idx'],
             torch.floor_divide(supervisions['start_frame'], subsampling),
             torch.floor_divide(supervisions['num_frames'], subsampling)),
            1).to(torch.int32)
        texts = supervisions['text']
        assert feature.ndim == 3

        feature = feature.to(device)
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        with torch.no_grad():
            nnet_output = model(feature)
        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2,
                                          1)  # now nnet_output is [N, T, C]

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
        assert LG.is_cuda()
        assert LG.device == device
        assert nnet_output.device == device
        # TODO(haowen): with a small `beam`, we may get empty `target_graph`,
        # thus `tot_scores` will be `inf`. Definitely we need to handle this later.
        lattices = k2.intersect_dense_pruned(LG, dense_fsa_vec, 2000.0, 20.0,
                                             30, 300)
        best_paths = k2.shortest_path(lattices)
        best_paths = best_paths.to('cpu')
        assert best_paths.shape[0] == len(texts)

        for i in range(len(texts)):
            hyp_words = [
                symbols.get(x) for x in best_paths[i].aux_labels if x > 0
            ]
            results.append((texts[i].split(' '), hyp_words))

        if batch_idx % 10 == 0:
            logging.info('Processed batch {}/{} ({:.6f}%)'.format(
                batch_idx, len(dataloader),
                float(batch_idx) / len(dataloader) * 100))

    return result


def main():
    # load L, G, symbol_table
    lang_dir = 'data/lang_nosp'
    symbol_table = k2.SymbolTable.from_file(lang_dir + '/words.txt')

    if not os.path.exists(lang_dir + '/LG.pt'):
        print("Loading L_disambig.fst.txt")
        with open(lang_dir + '/L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        print("Loading G.fsa.txt")
        with open(lang_dir + '/G.fsa.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=True)
        L = k2.arc_sort(L.invert_())
        G = k2.arc_sort(G)
        print("Intersecting L and G")
        LG = k2.intersect(L, G)
        print("TopSort L*G")
        LG = k2.top_sort(LG.invert_())
        print("Determinize L*G")
        LG = k2.determinize(LG)
        print("Remove disambiguation symbols on L*G")
        LG.labels[LG.labels >= 347] = 0
        LG.aux_labels[LG.aux_labels >= 200004] = 0
        LG = k2.add_epsilon_self_loops(LG)
        LG = k2.arc_sort(LG)
        print(k2.is_arc_sorted(k2.get_properties(LG)))
        torch.save(LG.as_dict(), lang_dir + '/LG.pt')
        print(LG)
    else:
        d = torch.load(lang_dir + '/LG.pt')
        print("Loading pre-prepared LG")
        LG = k2.Fsa.from_dict(d)

    return
    # load dataset
    feature_dir = 'exp/data'
    print("About to get test cuts")
    cuts_test = CutSet.from_json(feature_dir + '/cuts_test-clean.json.gz')

    print("About to create test dataset")
    test = K2SpeechRecognitionIterableDataset(cuts_test,
                                              max_frames=100000,
                                              shuffle=False)
    print("About to create test dataloader")
    test_dl = torch.utils.data.DataLoader(test, batch_size=None, num_workers=1)

    exp_dir = 'exp'
    setup_logger('{}/log/log-decode'.format(exp_dir))

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    print("About to load model")
    device_id = 1
    device = torch.device('cuda', device_id)
    model = Model(num_features=40, num_classes=364)
    checkpoint = os.path.join(exp_dir, 'epoch-9.pt')
    load_checkpoint(checkpoint, model)
    model.to(device)
    model.eval()

    LG = LG.to(device)
    LG.scores.requires_grad_(False)
    subsampling = 3  # must be kept in sync with model.
    results = decode(dataloader=test_dl,
                     model=model,
                     subsampling=subsampling,
                     device=device,
                     LG=LG,
                     symbols=symbol_table)
    for ref, hyp in results:
        print('ref=', ref, ', hyp=', hyp)
    # compute WER
    dist = sum(editdistance.eval(ref, hyp) for ref, hyp in results)
    total_words = sum(len(ref) for ref, _ in results)
    print('WER on {} sentences is {:.2f}%'.format(
        len(results),
        float(dist) / total_words * 100))
    logging.warning('Done')


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
