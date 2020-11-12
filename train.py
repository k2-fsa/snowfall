#!/usr/bin/env python3

import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import torchaudio
import torchaudio.models
import k2

from lhotse import CutSet, Fbank, LilcomFilesWriter, WavAugmenter
from lhotse.dataset import SpeechRecognitionDataset
from lhotse.dataset.speech_recognition import K2DataLoader, K2SpeechRecognitionDataset, \
    K2SpeechRecognitionIterableDataset, concat_cuts
from lhotse.recipes.librispeech import download_and_untar, prepare_librispeech, dataset_parts_full

from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from wav2letter import Wav2Letter


def create_decoding_graph(texts, graph, symbols):
    fsas = []
    for text in texts:
        filter_text = [
            i if i in symbols._sym2id else '<UNK>' for i in text.split(' ')
        ]
        word_ids = [symbols.get(i) for i in filter_text]
        fsa = k2.linear_fsa(word_ids)
        fsa = k2.arc_sort(fsa)
        decoding_graph = k2.intersect(fsa, graph).invert_()
        decoding_graph = k2.add_epsilon_self_loops(decoding_graph)
        fsas.append(decoding_graph)
    return k2.create_fsa_vec(fsas)


def get_objf(batch, model, device, graph, symbols, training, optimizer=None):
    feature = batch['features']
    supervisions = batch['supervisions']
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'], supervisions['start_frame'],
         supervisions['num_frames']), 1).to(torch.int32)
    texts = supervisions['text']
    assert feature.ndim == 3
    #print(feature.shape)
    #print(supervision_segments[:, 1] + supervision_segments[:, 2])

    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    feature = feature.to(device)
    if training:
        nnet_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output = model(feature)

    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

    # TODO(haowen): create decoding graph at the beginning of training
    decoding_graph = create_decoding_graph(texts, graph, symbols)
    decoding_graph.to_(device)
    decoding_graph.scores.requires_grad_(False)
    #print(nnet_output.shape)
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
    #dense_fsa_vec.scores.requires_grad_(True)
    assert decoding_graph.is_cuda()
    assert decoding_graph.device == device
    assert nnet_output.device == device
    #print(nnet_output.get_device())
    #print(dense_fsa_vec)
    target_graph = k2.intersect_dense_pruned(decoding_graph, dense_fsa_vec, 10,
                                             10000, 0)
    tot_scores = -k2.get_tot_scores(target_graph, True, False).sum()
    if training:
        optimizer.zero_grad()
        tot_scores.backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

    objf = tot_scores.detach().cpu()
    total_objf = objf.item()
    total_frames = nnet_output.shape[0]

    return total_objf, total_frames


def get_validation_objf(dataloader, model, device, graph, symbols):
    total_objf = 0.
    total_frames = 0.  # for display only

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        objf, frames = get_objf(batch, model, device, graph, symbols, False)
        total_objf += objf
        total_frames += frames

    return total_objf, total_frames


def train_one_epoch(dataloader, valid_dataloader, model, device, graph,
                    symbols, optimizer, current_epoch, num_epochs):
    total_objf = 0.
    total_frames = 0.

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        curr_batch_objf, curr_batch_frames = get_objf(batch, model, device,
                                                      graph, symbols, True,
                                                      optimizer)

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames

        if batch_idx % 100 == 0:
            logging.info(
                'processing batch {}, current epoch is {}/{} '
                'global average objf: {:.6f} over {} '
                'frames, current batch average objf: {:.6f} over {} frames'.
                format(
                    batch_idx,
                    current_epoch,
                    num_epochs,
                    total_objf / total_frames,
                    total_frames,
                    curr_batch_objf / curr_batch_frames,
                    curr_batch_frames,
                ))

        if valid_dataloader and batch_idx % 1000 == 0:
            total_valid_objf, total_valid_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
                graph=graph,
                symbols=symbols)
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames'.format(
                    total_valid_objf / total_valid_frames, total_valid_frames))
    return total_objf


def main():
    # load L, G, symbol_table
    lang_dir = 'data/lang_nosp'
    with open(lang_dir + '/L.fst.txt') as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False)

    with open(lang_dir + '/G.fsa.txt') as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=True)

    with open(lang_dir + '/words.txt') as f:
        symbol_table = k2.SymbolTable.from_str(f.read())

    L = k2.arc_sort(L.invert_())
    G = k2.arc_sort(G)
    graph = k2.intersect(L, G)
    graph = k2.arc_sort(graph)

    # load dataset
    feature_dir = 'exp/data1'
    cuts_train = CutSet.from_json(feature_dir +
                                  '/cuts_train-clean-100.json.gz')
   
    cuts_dev = CutSet.from_json(feature_dir + '/cuts_dev-clean.json.gz')

    train = K2SpeechRecognitionIterableDataset(cuts_train, shuffle=True)
    validate = K2SpeechRecognitionIterableDataset(cuts_dev, shuffle=False)
    train_dl = torch.utils.data.DataLoader(train,
                                           batch_size=None,
                                           num_workers=1)
    valid_dl = torch.utils.data.DataLoader(validate,
                                           batch_size=None,
                                           num_workers=1)

    dir = 'exp'
    setup_logger('{}/log/log-train'.format(dir))

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    device_id = 0
    device = torch.device('cuda', device_id)
    model = Wav2Letter(num_classes=364, input_type='mfcc', num_features=40)
    model.to(device)

    learning_rate = 0.001
    start_epoch = 0
    num_epochs = 10
    best_objf = 100000
    best_epoch = start_epoch
    best_model_path = os.path.join(dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(dir, 'best-epoch-info')

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=5e-4)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(start_epoch, num_epochs):
        curr_learning_rate = learning_rate * pow(0.4, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_learning_rate

        logging.info('epoch {}, learning rate {}'.format(
            epoch, curr_learning_rate))
        objf = train_one_epoch(dataloader=train_dl,
                               valid_dataloader=valid_dl,
                               model=model,
                               device=device,
                               graph=graph,
                               symbols=symbol_table,
                               optimizer=optimizer,
                               current_epoch=epoch,
                               num_epochs=num_epochs)
        if objf < best_objf:
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf)
            save_training_info(filename=best_epoch_info_filename,
                               model_path=best_model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=best_objf,
                               best_objf=best_objf,
                               best_epoch=best_epoch)

        # we always save the model for every epoch
        model_path = os.path.join(dir, 'epoch-{}.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf)
        epoch_info_filename = os.path.join(dir, 'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           best_epoch=best_epoch)

    logging.warning('Done')


if __name__ == '__main__':
    main()
