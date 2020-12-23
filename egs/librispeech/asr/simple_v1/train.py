#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
# Apache 2.0

import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import k2
import numpy as np
import torch
import torch.optim as optim

from lhotse import CutSet
from lhotse.dataset.speech_recognition import K2SpeechRecognitionIterableDataset
from lhotse.utils import fix_random_seed

from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from snowfall.common import save_checkpoint, load_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.models import AcousticModel
from snowfall.models.tdnn_lstm import TdnnLstm1b
from snowfall.training.ctc_graph import CtcTrainingGraphCompiler


def get_tot_objf_and_num_frames(tot_scores: torch.Tensor,
                                frames_per_seq: torch.Tensor
                                ) -> Tuple[float, int, int]:
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
        bad_indexes = torch.nonzero(~mask).squeeze(1)
        if bad_indexes.shape[0] > 0:
            print("Bad indexes: ", bad_indexes, ", bad lengths: ",
                  frames_per_seq[bad_indexes], " vs. max length ",
                  torch.max(frames_per_seq), ", avg ",
                  (torch.sum(frames_per_seq) / frames_per_seq.numel()))
    # print("finite_indexes = ", finite_indexes, ", tot_scores = ", tot_scores)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    return (tot_scores[finite_indexes].sum(), ok_frames, all_frames)


def get_objf(batch: Dict,
             model: AcousticModel,
             device: torch.device,
             graph_compiler: CtcTrainingGraphCompiler,
             training: bool,
             optimizer: Optional[torch.optim.Optimizer] = None):
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

    decoding_graph = graph_compiler.compile(texts).to(device)

    # nnet_output2 = nnet_output.clone()
    # blank_bias = -7.0
    # nnet_output2[:,:,0] += blank_bias

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
    assert decoding_graph.is_cuda()
    assert decoding_graph.device == device
    assert nnet_output.device == device
    # TODO(haowen): with a small `beam`, we may get empty `target_graph`,
    # thus `tot_scores` will be `inf`. Definitely we need to handle this later.
    target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec, 10.0)

    tot_scores = k2.get_tot_scores(target_graph,
                                   log_semiring=True,
                                   use_double_scores=True)

    (tot_score, tot_frames,
     all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                               supervision_segments[:, 2])

    if training:
        optimizer.zero_grad()
        (-tot_score).backward()
        clip_grad_value_(model.parameters(), 5.0)
        optimizer.step()

    ans = -tot_score.detach().cpu().item(), tot_frames.cpu().item(
    ), all_frames.cpu().item()
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel, device: torch.device,
                        graph_compiler: CtcTrainingGraphCompiler):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        objf, frames, all_frames = get_objf(batch, model, device,
                                            graph_compiler, False)
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel, device: torch.device,
                    graph_compiler: CtcTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int,
                    global_batch_idx_valid: int):
    total_objf, total_frames, total_all_frames = 0., 0., 0.
    time_waiting_for_batch = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()
        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = \
            get_objf(batch, model, device, graph_compiler, True, optimizer)

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, epoch {}/{} '
                'global average objf: {:.6f} over {} '
                'frames ({:.1f}% kept), current batch average objf: {:.6f} over {} frames ({:.1f}% kept) '
                'avg time waiting for batch {:.3f}s'.format(
                    batch_idx, current_epoch, num_epochs,
                    total_objf / total_frames, total_frames,
                    100.0 * total_frames / total_all_frames,
                    curr_batch_objf / (curr_batch_frames + 0.001),
                    curr_batch_frames,
                    100.0 * curr_batch_frames / curr_batch_all_frames,
                    time_waiting_for_batch / max(1, batch_idx)))

            tb_writer.add_scalar('train/global_average_objf',
                                 total_objf / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_objf',
                                 curr_batch_objf / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)
            # if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 200 == 0:
            total_valid_objf, total_valid_frames, total_valid_all_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
                graph_compiler=graph_compiler)
            global_batch_idx_valid += 1
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'
                    .format(total_valid_objf / total_valid_frames,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            tb_writer.add_scalar('train/global_valid_average_objf',
                             total_valid_objf / total_valid_frames,
                             global_batch_idx_valid)
        prev_timestamp = datetime.now()
    return total_objf / total_frames


def describe(model: nn.Module):
    print('=' * 80)
    print('Model parameters summary:')
    print('=' * 80)
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        print(f'* {name}: {num_params:>{80 - len(name) - 4}}')
    print('=' * 80)
    print('Total:', total)
    print('=' * 80)


def main():
    fix_random_seed(42)

    exp_dir = 'exp-lstm-adam'
    setup_logger('{}/log/log-train'.format(exp_dir))
    tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    word_symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')

    logging.info("Loading L.fst")
    if (lang_dir / 'Linv.pt').exists():
        L_inv = k2.Fsa.from_dict(torch.load(lang_dir / 'Linv.pt'))
    else:
        with open(lang_dir / 'L.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
            L_inv = k2.arc_sort(L.invert_())
            torch.save(L_inv.as_dict(), lang_dir / 'Linv.pt')

    graph_compiler = CtcTrainingGraphCompiler(
        L_inv=L_inv,
        phones=phone_symbol_table,
        words=word_symbol_table
    )

    # load dataset
    feature_dir = Path('exp/data')
    logging.info("About to get train cuts")
    cuts_train = CutSet.from_json(feature_dir /
                                  'cuts_train-clean-100.json.gz')
    logging.info("About to get dev cuts")
    cuts_dev = CutSet.from_json(feature_dir / 'cuts_dev-clean.json.gz')

    logging.info("About to create train dataset")
    train = K2SpeechRecognitionIterableDataset(cuts_train,
                                               max_frames=90000,
                                               shuffle=True)
    logging.info("About to create dev dataset")
    validate = K2SpeechRecognitionIterableDataset(cuts_dev,
                                                  max_frames=90000,
                                                  shuffle=False,
                                                  concat_cuts=False)
    logging.info("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(train,
                                           batch_size=None,
                                           num_workers=4)
    logging.info("About to create dev dataloader")
    valid_dl = torch.utils.data.DataLoader(validate,
                                           batch_size=None,
                                           num_workers=1)

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    logging.info("About to create model")
    device_id = 0
    device = torch.device('cuda', device_id)
    model = TdnnLstm1b(num_features=40, num_classes=len(phone_symbol_table), subsampling_factor=3)

    learning_rate = 0.00001
    start_epoch = 0
    num_epochs = 8
    best_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0 # for logging only
    global_batch_idx_valid = 0 # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        (epoch, learning_rate, objf) = load_checkpoint(filename=model_path, model=model)
        best_objf = objf
        logging.info("epoch = {}, objf = {}".format(epoch, objf))

    model.to(device)
    describe(model)

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=learning_rate,
    #                       momentum=0.9,
    #                       weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(),
                            # lr=learning_rate,
                            weight_decay=5e-4)

    for epoch in range(start_epoch, num_epochs):
        curr_learning_rate = 1e-3
        # curr_learning_rate = learning_rate * pow(0.4, epoch)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = curr_learning_rate

        tb_writer.add_scalar('learning_rate', curr_learning_rate, epoch)

        logging.info('epoch {}, learning rate {}'.format(
            epoch, curr_learning_rate))
        objf = train_one_epoch(dataloader=train_dl,
                               valid_dataloader=valid_dl,
                               model=model,
                               device=device,
                               graph_compiler=graph_compiler,
                               optimizer=optimizer,
                               current_epoch=epoch,
                               tb_writer=tb_writer,
                               num_epochs=num_epochs,
                               global_batch_idx_train=global_batch_idx_train,
                               global_batch_idx_valid=global_batch_idx_valid)
        # the lower, the better
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
