#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import k2
import logging
import math
import numpy as np
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, CutMix, K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.utils import fix_random_seed
from snowfall.common import describe
from snowfall.common import get_phone_symbols
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.models import AcousticModel
from snowfall.models.transformer import Noam, Transformer
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
             is_training: bool,
             is_update: bool,
             accum_grad: int = 1,
             att_rate: float = 0.0,
             optimizer: Optional[torch.optim.Optimizer] = None):
    feature = batch['features']
    supervisions = batch['supervisions']
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         torch.floor_divide(supervisions['start_frame'],
                            model.subsampling_factor),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    texts = supervisions['text']
    texts = [texts[idx] for idx in indices]
    assert feature.ndim == 3
    # print(supervision_segments[:, 1] + supervision_segments[:, 2])

    feature = feature.to(device)
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    if is_training:
        nnet_output, encoder_memory, memory_mask = model(feature, supervision_segments)
        if att_rate != 0.0:
            att_loss = model.decoder_forward(encoder_memory, memory_mask, supervisions, graph_compiler)
    else:
        with torch.no_grad():
            nnet_output, encoder_memory, memory_mask = model(feature, supervision_segments)
            if att_rate != 0.0:
                att_loss = model.decoder_forward(encoder_memory, memory_mask, supervisions, graph_compiler)

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

    target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec, 10.0)

    tot_scores = target_graph.get_tot_scores(
        log_semiring=True,
        use_double_scores=True)

    (tot_score, tot_frames,
     all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                               supervision_segments[:, 2])

    if is_training:
        if att_rate != 0.0:
            loss = (- (1.0 - att_rate) * tot_score + att_rate * att_loss) / (len(texts) * accum_grad)
        else:
            loss = (-tot_score) / (len(texts) * accum_grad)
        loss.backward()
        if is_update:
            clip_grad_value_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()

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
        objf, frames, all_frames = get_objf(
            batch=batch,
            model=model,
            device=device,
            graph_compiler=graph_compiler,
            is_training=False,
            is_update=False,
        )
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel, device: torch.device,
                    graph_compiler: CtcTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    accum_grad: int,
                    att_rate: float,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int):
    """One epoch training and validation.

    Args:
        dataloader: Training dataloader
        valid_dataloader: Validation dataloader
        model: Acoustic model to be trained
        device: Training device, torch.device("cpu") or torch.device("cuda", device_id)
        graph_compiler: MMI training graph compiler
        optimizer: Training optimizer
        accum_grad: Number of gradient accumulation
        att_rate: Attention loss rate, final loss is att_rate * att_loss + (1-att_rate) * other_loss
        current_epoch: current training epoch, for logging only
        tb_writer: tensorboard SummaryWriter
        num_epochs: total number of training epochs, for logging only
        global_batch_idx_train: global training batch index before this epoch, for logging only

    Returns:
        A tuple of 3 scalar:  (total_objf / total_frames, valid_average_objf, global_batch_idx_train)
        - `total_objf / total_frames` is the average training loss
        - `valid_average_objf` is the average validation loss
        - `global_batch_idx_train` is the global training batch index after this epoch
    """
    total_objf, total_frames, total_all_frames = 0., 0., 0.
    valid_average_objf = float('inf')
    time_waiting_for_batch = 0
    forward_count = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        forward_count += 1
        if forward_count == accum_grad:
            is_update = True
            forward_count = 0
        else:
            is_update = False

        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()
        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = get_objf(
            batch=batch,
            model=model,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            is_update=is_update,
            accum_grad=accum_grad,
            att_rate=att_rate,
            optimizer=optimizer
        )

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
            valid_average_objf = total_valid_objf / total_valid_frames
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'
                    .format(valid_average_objf,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            tb_writer.add_scalar('train/global_valid_average_objf',
                                 valid_average_objf,
                                 global_batch_idx_train)
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help="Number of traning epochs.")
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help="Number of start epoch.")
    parser.add_argument(
        '--max-frames',
        type=int,
        default=60000,
        help="Maximum number of feature frames in a single batch.")
    parser.add_argument(
        '--accum-grad',
        type=int,
        default=1,
        help="Number of gradient accumulation.")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    return parser


def main():
    args = get_parser().parse_args()

    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    max_frames = args.max_frames
    accum_grad = args.accum_grad
    att_rate = args.att_rate

    fix_random_seed(42)

    exp_dir = Path('exp-transformer-noam-ctc-att-musan')
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
    phone_ids = get_phone_symbols(phone_symbol_table)

    # load dataset
    feature_dir = Path('exp/data')
    logging.info("About to get train cuts")
    cuts_train = CutSet.from_json(feature_dir /
                                  'cuts_train-clean-100.json.gz')
    logging.info("About to get dev cuts")
    cuts_dev = CutSet.from_json(feature_dir / 'cuts_dev-clean.json.gz')
    logging.info("About to get Musan cuts")
    cuts_musan = CutSet.from_json(feature_dir / 'cuts_musan.json.gz')

    logging.info("About to create train dataset")
    train = K2SpeechRecognitionDataset(
        cuts_train,
        cut_transforms=[
            CutConcatenate(),
            CutMix(
                cuts=cuts_musan,
                prob=0.5,
                snr=(10, 20)
            )
        ]
    )
    train_sampler = SingleCutSampler(
        cuts_train,
        max_frames=max_frames,
        shuffle=True,
    )
    logging.info("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=4
    )
    logging.info("About to create dev dataset")
    validate = K2SpeechRecognitionDataset(cuts_dev)
    valid_sampler = SingleCutSampler(cuts_dev, max_frames=max_frames)
    logging.info("About to create dev dataloader")
    valid_dl = torch.utils.data.DataLoader(
        validate,
        sampler=valid_sampler,
        batch_size=None,
        num_workers=1
    )

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    logging.info("About to create model")
    device_id = 0
    device = torch.device('cuda', device_id)

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    model = Transformer(
        num_features=40,
        num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
        subsampling_factor=4,
        num_decoder_layers=num_decoder_layers)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    model.to(device)
    describe(model)

    optimizer = Noam(model.parameters(),
                     model_size=256,
                     factor=5.0,
                     warm_step=25000)

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        curr_learning_rate = optimizer._rate

        tb_writer.add_scalar('learning_rate', curr_learning_rate, epoch)

        logging.info('epoch {}, learning rate {}'.format(
            epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(dataloader=train_dl,
                                                                   valid_dataloader=valid_dl,
                                                                   model=model,
                                                                   device=device,
                                                                   graph_compiler=graph_compiler,
                                                                   optimizer=optimizer,
                                                                   accum_grad=accum_grad,
                                                                   att_rate=att_rate,
                                                                   current_epoch=epoch,
                                                                   tb_writer=tb_writer,
                                                                   num_epochs=num_epochs,
                                                                   global_batch_idx_train=global_batch_idx_train)
        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            valid_objf=valid_objf,
                            global_batch_idx_train=global_batch_idx_train)
            save_training_info(filename=best_epoch_info_filename,
                               model_path=best_model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=best_objf,
                               best_objf=best_objf,
                               valid_objf=valid_objf,
                               best_valid_objf=best_valid_objf,
                               best_epoch=best_epoch)

        # we always save the model for every epoch
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        valid_objf=valid_objf,
                        global_batch_idx_train=global_batch_idx_train)
        epoch_info_filename = os.path.join(exp_dir,
                                           'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           valid_objf=valid_objf,
                           best_valid_objf=best_valid_objf,
                           best_epoch=best_epoch)

    logging.warning('Done')


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
