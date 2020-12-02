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
from model import Model


def create_decoding_graph(texts, L, symbols):
    word_ids_list = []
    for text in texts:
        filter_text = [
            i if i in symbols._sym2id else '<UNK>' for i in text.split(' ')
        ]
        word_ids = [symbols.get(i) for i in filter_text]
        word_ids_list.append(word_ids)
    fsa = k2.linear_fsa(word_ids_list)
    decoding_graph = k2.intersect(fsa, L).invert_()
    decoding_graph = k2.add_epsilon_self_loops(decoding_graph)
    return decoding_graph

def get_tot_objf_and_num_frames(tot_scores, frames_per_seq):
    ''' Figures out the total score(log-prob) over all successful supervision segments
    (i.e. those for which the total score wasn't -infinity), and the corresponding
    number of frames of neural net output
         Args:
            tot_scores: a Torch tensor of shape (num_segments,) containing total scores
                       from forward-backward
        frames_per_seq: a Torch tensor of shape (num_segments,) containing the number of
                       frames for each segment
        Returns:
             Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames)
        where ok_frames is the frames for successful (finite) segments, and
       all_frames is the frames for all segments (finite or not).
    '''
    mask = torch.ne(tot_scores, -math.inf)
    # finite_indexes is a tensor containing successful segment indexes, e.g.
    # [ 0 1 3 4 5 ]
    finite_indexes = torch.nonzero(mask).squeeze(1)
    if False:
        bad_indexes =  torch.nonzero(~mask).squeeze(1)
        if bad_indexes.shape[0] > 0:
            print("Bad indexes: ", bad_indexes, ", bad lengths: ", frames_per_seq[bad_indexes],
                  " vs. max length ", torch.max(frames_per_seq), ", avg ",
                  (torch.sum(frames_per_seq) / frames_per_seq.numel()))
    #print("finite_indexes = ", finite_indexes, ", tot_scores = ", tot_scores)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    return (tot_scores[finite_indexes].sum(), ok_frames, all_frames)



def get_objf(batch, model, subsampling, device, L, symbols, training, optimizer=None):
    feature = batch['features']
    supervisions = batch['supervisions']
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         torch.floor_divide(supervisions['start_frame'], subsampling),
         torch.floor_divide(supervisions['num_frames'], subsampling)), 1).to(torch.int32)
    _,indices = torch.sort(-supervision_segments[:,2])
    supervision_segments = supervision_segments[indices,:]

    texts = supervisions['text']
    assert feature.ndim == 3
    # print(supervision_segments[:, 1] + supervision_segments[:, 2])

    feature = feature.to(device)
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    if training:
        nnet_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output = model(feature)

    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]


    # TODO(haowen): create decoding graph (and cache) at the beginning of training
    decoding_graph = create_decoding_graph(texts, L, symbols).to(device)
    decoding_graph.scores.requires_grad_(False)

    #nnet_output2 = nnet_output.clone()
    #blank_bias = -7.0
    #nnet_output2[:,:,0] += blank_bias

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
    assert decoding_graph.is_cuda()
    assert decoding_graph.device == device
    assert nnet_output.device == device
    # TODO(haowen): with a small `beam`, we may get empty `target_graph`,
    # thus `tot_scores` will be `inf`. Definitely we need to handle this later.
    target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec, 10.0)

    tot_scores = k2.get_tot_scores(target_graph, True, False)

    (tot_score, tot_frames, all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                                          supervision_segments[:, 2])

    if training:
        optimizer.zero_grad()
        (-tot_score).backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

    ans = -tot_score.detach().cpu().item(), tot_frames.cpu().item(), all_frames.cpu().item()
    return  ans


def get_validation_objf(dataloader, model, subsampling, device, L, symbols):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        objf, frames, all_frames = get_objf(batch, model, subsampling,
                                            device, L, symbols, False)
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader, valid_dataloader, model,
                    subsampling, device, L, symbols,
                    optimizer, current_epoch, num_epochs):
    total_objf, total_frames, total_all_frames = 0., 0., 0.

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = \
          get_objf(batch, model, subsampling, device, L, symbols, True, optimizer)

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0:
            logging.info(
                'processing batch {}, current epoch is {}/{} '
                'global average objf: {:.6f} over {} '
                'frames ({:.1f}% kept), current batch average objf: {:.6f} over {} frames ({:.1f}% kept) '.
                format(
                    batch_idx,
                    current_epoch,
                    num_epochs,
                    total_objf / total_frames,
                    total_frames,
                    100.0 * total_frames / total_all_frames,
                    curr_batch_objf / (curr_batch_frames+0.001),
                    curr_batch_frames,
                    100.0 * curr_batch_frames / curr_batch_all_frames))
            #if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 200 == 0:
            total_valid_objf, total_valid_frames, total_valid_all_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                subsampling=subsampling,
                device=device,
                L=L,
                symbols=symbols)
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'.format(
                    total_valid_objf / total_valid_frames, total_valid_frames,
                    100.0 * total_valid_frames / total_valid_all_frames))
    return total_objf


def main():
    # load L, G, symbol_table
    lang_dir = 'data/lang_nosp'
    symbol_table = k2.SymbolTable.from_file(lang_dir + '/words.txt')

    ## This commented code created LG.  We don't need that there.
    ## There were problems with disambiguation symbols; the G has
    ## disambiguation symbols which L.fst doesn't support.
    # if not os.path.exists(lang_dir + '/LG.pt'):
    #     print("Loading L.fst.txt")
    #     with open(lang_dir + '/L.fst.txt') as f:
    #         L = k2.Fsa.from_openfst(f.read(), acceptor=False)
    #     print("Loading G.fsa.txt")
    #     with open(lang_dir + '/G.fsa.txt') as f:
    #         G = k2.Fsa.from_openfst(f.read(), acceptor=True)
    #     print("Arc-sorting L...")
    #     L = k2.arc_sort(L.invert_())
    #     G = k2.arc_sort(G)
    #     print(k2.is_arc_sorted(k2.get_properties(L)))
    #     print(k2.is_arc_sorted(k2.get_properties(G)))
    #     print("Intersecting L and G")
    #     graph = k2.intersect(L, G)
    #     graph = k2.arc_sort(graph)
    #     print(k2.is_arc_sorted(k2.get_properties(graph)))
    #     torch.save(graph.as_dict(), lang_dir + '/LG.pt')
    # else:
    #     d = torch.load(lang_dir + '/LG.pt')
    #     print("Loading pre-prepared LG")
    #     graph = k2.Fsa.from_dict(d)

    print("Loading L.fst")
    if os.path.exists(lang_dir + '/Linv.pt'):
        L = k2.Fsa.from_dict(torch.load(lang_dir + '/Linv.pt'))
    else:
        with open(lang_dir + '/L.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
            L = k2.arc_sort(L.invert_())
            torch.save(L.as_dict(), lang_dir + '/Linv.pt')

    # load dataset
    feature_dir = 'exp/data'
    print("About to get train cuts")
    cuts_train = CutSet.from_json(feature_dir +
                                  '/cuts_train-clean-100.json.gz')
    print("About to get dev cuts")
    cuts_dev = CutSet.from_json(feature_dir + '/cuts_dev-clean.json.gz')

    print("About to create train dataset")
    train = K2SpeechRecognitionIterableDataset(cuts_train, max_frames=100000, shuffle=True)
    print("About to create dev dataset")
    validate = K2SpeechRecognitionIterableDataset(cuts_dev, max_frames=100000, shuffle=False)
    print("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(train,
                                           batch_size=None,
                                           num_workers=1)
    print("About to create dev dataloader")
    valid_dl = torch.utils.data.DataLoader(validate,
                                           batch_size=None,
                                           num_workers=1)

    exp_dir = 'exp'
    setup_logger('{}/log/log-train'.format(exp_dir))

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    print("About to create model")
    device_id = 0
    device = torch.device('cuda', device_id)
    model = Model(num_features=40, num_classes=364)
    model.to(device)

    learning_rate = 0.00001
    start_epoch = 0
    num_epochs = 10
    best_objf = 100000
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')


    if False:
        filename = 'exp/epoch-0.pt'
        (_, _, objf) = load_checkpoint(filename,
                                       model)

    #optimizer = optim.Adam(model.parameters(),
    #                       lr=learning_rate,
    #                       weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=5e-4)
    subsampling = 3 # must be kept in sync with model.


    for epoch in range(start_epoch, num_epochs):
        curr_learning_rate = learning_rate * pow(0.4, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_learning_rate

        logging.info('epoch {}, learning rate {}'.format(
            epoch, curr_learning_rate))
        objf = train_one_epoch(dataloader=train_dl,
                               valid_dataloader=valid_dl,
                               model=model,
                               subsampling=subsampling,
                               device=device,
                               L=L,
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
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf)
        epoch_info_filename = os.path.join(exp_dir,
                                           'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           best_epoch=best_epoch)

    logging.warning('Done')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
