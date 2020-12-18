#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
# Apache 2.0

import logging
import os
from pathlib import Path
from typing import Union

import k2
import torch
from k2 import Fsa, SymbolTable
from kaldialign import edit_distance
from lhotse import CutSet
from lhotse.dataset.speech_recognition import K2SpeechRecognitionIterableDataset

from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.decoding.graph import compile_LG
from snowfall.models import AcousticModel
from snowfall.models.tdnn import Tdnn1a
from snowfall.models.tdnn_lstm import TdnnLstm1b


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
        texts = [texts[idx] for idx in indices]
        assert feature.ndim == 3

        feature = feature.to(device)
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        with torch.no_grad():
            nnet_output = model(feature)
        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2,
                                          1)  # now nnet_output is [N, T, C]

        blank_bias = -3.0
        nnet_output[:, :, 0] += blank_bias

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
        best_paths = best_paths.to('cpu')
        assert best_paths.shape[0] == len(texts)

        for i in range(len(texts)):
            if isinstance(best_paths[i].aux_labels, torch.Tensor):
                aux_labels = best_paths[i].aux_labels
            else:
                # it's a ragged tensor
                aux_labels = best_paths[i].aux_labels.values()
            aux_labels = aux_labels[aux_labels > 0]
            aux_labels = aux_labels.tolist()
            hyp_words = [symbols.get(x) for x in aux_labels]
            results.append((texts[i].split(' '), hyp_words))

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, cuts processed until now is {}/{} ({:.6f}%)'.format(
                    batch_idx, num_cuts, tot_num_cuts,
                    float(num_cuts) / tot_num_cuts * 100))

        num_cuts += len(texts)

    return results


def find_first_disambig_symbol(symbols: SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))


def main():
    exp_dir = Path('exp-lstm-adam')
    setup_logger('{}/log/log-decode'.format(exp_dir))

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')

    if not os.path.exists(lang_dir / 'LG.pt'):
        print("Loading L_disambig.fst.txt")
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        print("Loading G.fsa.txt")
        with open(lang_dir / 'G.fsa.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=True)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        LG = compile_LG(L=L,
                        G=G,
                        labels_disambig_id_start=first_phone_disambig_id,
                        aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(LG.as_dict(), lang_dir / 'LG.pt')
    else:
        # TODO(haowen): support save raggedInt
        print("Loading pre-compiled LG")
        d = torch.load(lang_dir / 'LG.pt')
        LG = k2.Fsa.from_dict(d)

    # load dataset
    feature_dir = Path('exp/data')
    print("About to get test cuts")
    cuts_test = CutSet.from_json(feature_dir / 'cuts_test-clean.json.gz')

    print("About to create test dataset")
    test = K2SpeechRecognitionIterableDataset(cuts_test,
                                              max_frames=50000,
                                              shuffle=False,
                                              concat_cuts=False)
    print("About to create test dataloader")
    test_dl = torch.utils.data.DataLoader(test, batch_size=None, num_workers=1)

    #  if not torch.cuda.is_available():
    #  logging.error('No GPU detected!')
    #  sys.exit(-1)

    print("About to load model")
    # Note: Use "export CUDA_VISIBLE_DEVICES=N" to setup device id to N
    # device = torch.device('cuda', 1)
    device = torch.device('cuda')
    model = TdnnLstm1b(num_features=40, num_classes=len(phone_symbol_table))
    checkpoint = os.path.join(exp_dir, 'epoch-8.pt')
    load_checkpoint(checkpoint, model)
    model.to(device)
    model.eval()

    print("convert LG to device")
    LG = LG.to(device)
    LG.requires_grad_(False)
    print("About to decode")
    results = decode(dataloader=test_dl,
                     model=model,
                     device=device,
                     LG=LG,
                     symbols=symbol_table)
    for ref, hyp in results:
        print('ref=', ref)
        print('hyp=', hyp)
    # compute WER
    dists = [edit_distance(r, h) for r, h in results]
    errors = {
        key: sum(dist[key] for dist in dists)
        for key in ['sub', 'ins', 'del', 'total']
    }
    total_words = sum(len(ref) for ref, _ in results)
    # Print Kaldi-like message:
    # %WER 8.20 [ 4459 / 54402, 695 ins, 427 del, 3337 sub ]
    print(
        f'%WER {errors["total"] / total_words:.2%} '
        f'[{errors["total"]} / {total_words}, {errors["ins"]} ins, {errors["del"]} del, {errors["sub"]} sub ]'
    )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
