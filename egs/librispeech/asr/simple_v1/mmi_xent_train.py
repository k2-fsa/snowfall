#!/usr/bin/env python3
# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
# Apache 2.0

import k2
import logging
import math
import numpy as np
import os
import argparse
import sys
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.dataset import CutConcatenate, CutMix, K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.utils import fix_random_seed
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.common import describe
from snowfall.models import AcousticModel
from snowfall.models.tdnn_lstm import TdnnLstm1c
from snowfall.training.diagnostics import measure_gradient_norms, optim_step_and_measure_param_change
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols


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
             P: k2.Fsa,
             device: torch.device,
             graph_compiler: MmiTrainingGraphCompiler,
             is_training: bool,
             den_scale: float = 1.0,
             xent_scale: float = 0.0,
             smooth_scale: float = 0.0,
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
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
    if is_training:
        nnet_output, xent_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output, xent_output = model(feature)

    # nnet_output and xent_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]
    xent_output = xent_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

    if is_training:
        num, den = graph_compiler.compile(texts, P)
    else:
        with torch.no_grad():
            num, den = graph_compiler.compile(texts, P)

    assert num.requires_grad == is_training
    assert den.requires_grad is False
    num = num.to(device)
    den = den.to(device)

    # nnet_output2 = nnet_output.clone()
    # blank_bias = -7.0
    # nnet_output2[:,:,0] += blank_bias

    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)
    assert nnet_output.device == device

    num = k2.intersect_dense(num,
                             dense_fsa_vec,
                             10.0,
                             seqframe_idx_name='seqframe_idx')
    den = k2.intersect_dense(den, dense_fsa_vec, 10.0)

    num_tot_scores = num.get_tot_scores(
        log_semiring=True,
        use_double_scores=True)
    den_tot_scores = den.get_tot_scores(
        log_semiring=True,
        use_double_scores=True)
    tot_scores = num_tot_scores - den_scale * den_tot_scores

    (tot_score, tot_frames,
     all_frames) = get_tot_objf_and_num_frames(tot_scores,
                                               supervision_segments[:, 2])
    num_rows = dense_fsa_vec.scores.shape[0]
    num_cols = dense_fsa_vec.scores.shape[1] - 1
    num_post_sparse = k2.create_sparse(rows=num.seqframe_idx,
                                      cols=num.phones,
                                      values=num.get_arc_post(True,
                                                                   True).exp(),
                                      size=(num_rows, num_cols),
                                      min_col_index=0)
    num_post_mat = num_post_sparse.to_dense()

    if False:
        num_post_mat = num_post_mat.detach()

    xent_output = masked_nnet_output(xent_output, supervision_segments)

    xent_loss = - (xent_output * num_post_mat).sum()

    phone_counts = num_post_mat.sum(0)
    smooth_loss = - (phone_counts * torch.log(phone_counts/phone_counts.sum() + 1.0e-20)).sum()

    final_loss = (- tot_score) + xent_scale * xent_loss - smooth_scale * smooth_loss

    if is_training:
        def maybe_log_gradients(tag: str):
            if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
                tb_writer.add_scalars(
                    tag,
                    measure_gradient_norms(model, norm='l1'),
                    global_step=global_batch_idx_train
                )

        optimizer.zero_grad()
        (final_loss).backward()
        maybe_log_gradients('train/grad_norms')
        clip_grad_value_(model.parameters(), 5.0)
        maybe_log_gradients('train/clipped_grad_norms')
        if global_batch_idx_train % 200 == 0:
            # Once in a time we will perform a more costly diagnostic
            # to check the relative parameter change per minibatch.
            deltas = optim_step_and_measure_param_change(model, optimizer)
            tb_writer.add_scalars(
                'train/relative_param_change_per_minibatch',
                deltas,
                global_step=global_batch_idx_train
            )
        else:
            optimizer.step()

    ans = (
        -tot_score.detach().cpu().item(),
        (xent_scale * xent_loss).detach().cpu().item(),
        (smooth_scale * smooth_loss).detach().cpu().item(),
        tot_frames.cpu().item(),
        all_frames.cpu().item()
    )
    return ans


def masked_nnet_output(log_probs: torch.Tensor,
                supervision_segments: torch.Tensor) -> torch.Tensor:
    '''Get masked neural net log-softmax outputs without padding 
    like DenseFsaVec construction.

    Args:
        log_probs:
        A 3-D tensor of dtype `torch.float32` with shape `(N, T, C)`,
        where `N` is the number of sequences, `T` the maximum input
        length, and `C` the number of output classes.
        supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.

        Note:
            - `0 < start_frame + duration <= T`
            - `0 <= start_frame < T`
            - `duration > 0`

        Caution:
            The last column, i.e., the duration column, has to be sorted
            in **decreasing** order. That is, the first supervision_segment
            (the first row) has the largest duration.
    '''
    assert log_probs.ndim == 3
    assert log_probs.dtype == torch.float32
    assert supervision_segments.ndim == 2
    assert supervision_segments.dtype == torch.int32
    assert supervision_segments.device.type == 'cpu'

    N, T, C = log_probs.shape

    # Also, if a particular FSA has T frames of neural net output,
    # we actually have T+1 potential indexes, 0 through T, so there is
    # space for the terminating final-symbol on frame T.  (On the last
    # frame, each symbol have logprob=0).
    placeholder = torch.tensor([0])  # this extra row is for the last frame
    indexes = []
    last_frame_indexes = []
    cur = 0
    for segment in supervision_segments:
        segment_index, start_frame, duration = segment.tolist()
        assert 0 <= segment_index < N
        assert 0 <= start_frame < T
        assert duration > 0
        assert start_frame + duration <= T
        offset = segment_index * T
        indexes.append(
            torch.arange(start_frame, start_frame + duration) + offset)
        indexes.append(placeholder)
        cur += duration
        last_frame_indexes.append(cur)
        cur += 1  # increment for the extra row

    device = log_probs.device
    indexes = torch.cat(indexes).to(device)

    scores = torch.empty(cur, C, dtype=log_probs.dtype, device=device)
    scores[:, :] = log_probs.reshape(-1, C).index_select(0, indexes)

    scores = log_probs.reshape(-1, C).index_select(0, indexes)
    scores[last_frame_indexes] = torch.tensor([0.0] * C,
                                                  device=device)

    return scores

def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        P: k2.Fsa,
                        device: torch.device,
                        graph_compiler: MmiTrainingGraphCompiler,
                        den_scale: float = 1.0,
                        xent_scale: float = 0.0,
                        smooth_scale: float = 0.0):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    total_loss = 0.
    total_mmi_loss = 0.
    total_xent_loss = 0.
    total_smooth_loss = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        mmi_loss, xent_loss, smooth_loss, frames, all_frames = get_objf(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=False,
            den_scale=den_scale,
            xent_scale=xent_scale,
            smooth_scale=smooth_scale
        )

        cur_loss = mmi_loss + xent_loss + smooth_loss
        total_loss += cur_loss
        total_mmi_loss += mmi_loss
        total_xent_loss += xent_loss
        total_smooth_loss += smooth_loss
        total_frames += frames
        total_all_frames += all_frames

    return total_loss, total_mmi_loss, total_xent_loss, total_smooth_loss, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel, P: k2.Fsa,
                    device: torch.device,
                    graph_compiler: MmiTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    den_scale: float,
                    xent_scale: float,
                    smooth_scale: float,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int):
    total_loss, total_mmi_loss, total_xent_loss, total_smooth_loss, total_frames, total_all_frames = 0., 0., 0., 0., 0., 0.
    valid_average_loss = float('inf')
    time_waiting_for_batch = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()

        P.set_scores_stochastic_(model.P_scores)
        assert P.is_cpu
        assert P.requires_grad is True

        curr_batch_mmi_loss, curr_batch_xent_loss, curr_batch_smooth_loss, curr_batch_frames, curr_batch_all_frames = get_objf(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            den_scale=den_scale,
            xent_scale=xent_scale,
            smooth_scale=smooth_scale,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
            optimizer=optimizer
        )

        total_mmi_loss += curr_batch_mmi_loss
        total_xent_loss += curr_batch_xent_loss
        total_smooth_loss += curr_batch_smooth_loss
        curr_batch_loss = curr_batch_mmi_loss + curr_batch_xent_loss + curr_batch_smooth_loss
        total_loss += curr_batch_loss
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, epoch {}/{} '
                'global average loss: {:.6f}, '
                'global average mmi loss: {:.6f}, '
                'global average xent loss: {:.6f} over {}, '
                'global average smooth loss: {:.6f} over {} '
                'frames ({:.1f}% kept), '
                'current batch average loss: {:.6f}, '
                'current batch average mmi loss: {:.6f}, '
                'current batch average xent loss: {:.6f}, '
                'current batch average smooth loss: {:.6f} '
                'over {} frames ({:.1f}% kept) '
                'avg time waiting for batch {:.3f}s'.format(
                    batch_idx, current_epoch, num_epochs,
                    total_loss / total_frames,
                    total_mmi_loss / total_frames,
                    total_xent_loss / total_frames, total_frames,
                    total_smooth_loss / total_frames, total_frames,
                    100.0 * total_frames / total_all_frames,
                    curr_batch_loss / (curr_batch_frames + 0.001),
                    curr_batch_mmi_loss / (curr_batch_frames + 0.001),
                    curr_batch_xent_loss / (curr_batch_frames + 0.001),
                    curr_batch_smooth_loss / (curr_batch_frames + 0.001),
                    curr_batch_frames,
                    100.0 * curr_batch_frames / curr_batch_all_frames,
                    time_waiting_for_batch / max(1, batch_idx)))

            tb_writer.add_scalar('train/global_average_loss',
                                 total_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/global_average_mmi_loss',
                                 total_mmi_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/global_average_xent_loss',
                                 total_xent_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/global_average_smooth_loss',
                                 total_smooth_loss / total_frames, global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_loss',
                                 curr_batch_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_mmi_loss',
                                 curr_batch_mmi_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_xent_loss',
                                 curr_batch_xent_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            tb_writer.add_scalar('train/current_batch_average_smooth_loss',
                                 curr_batch_smooth_loss / (curr_batch_frames + 0.001),
                                 global_batch_idx_train)

            # if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 200 == 0:
            total_valid_loss, total_valid_mmi_loss, total_valid_xent_loss, \
                total_valid_smooth_loss, total_valid_frames, total_valid_all_frames= get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                P=P,
                device=device,
                graph_compiler=graph_compiler,
                den_scale=den_scale,
                xent_scale=xent_scale,
                smooth_scale=smooth_scale)
            valid_average_loss = total_valid_loss / total_valid_frames
            model.train()
            logging.info(
                'Validation average loss: {:.6f}, '
                'Validation average mmi loss: {:.6f}, '
                'Validation average xent loss: {:.6f}, '
                'Validation average smooth loss: {:.6f} '
                'over {} frames ({:.1f}% kept)'
                    .format(total_valid_loss / total_valid_frames,
                            total_valid_mmi_loss / total_valid_frames,
                            total_valid_xent_loss / total_valid_frames,
                            total_valid_smooth_loss / total_valid_frames,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            tb_writer.add_scalar('train/global_valid_average_loss',
                             total_valid_loss / total_valid_frames,
                             global_batch_idx_train)

            tb_writer.add_scalar('train/global_valid_average_mmi_loss',
                             total_valid_mmi_loss / total_valid_frames,
                             global_batch_idx_train)

            tb_writer.add_scalar('train/global_valid_average_xent_loss',
                             total_valid_xent_loss / total_valid_frames,
                             global_batch_idx_train)

            tb_writer.add_scalar('train/global_valid_average_smooth_loss',
                             total_valid_smooth_loss / total_valid_frames,
                             global_batch_idx_train)

            model.write_tensorboard_diagnostics(tb_writer, global_step=global_batch_idx_train)
        prev_timestamp = datetime.now()
    return valid_average_loss / total_frames, valid_average_loss, global_batch_idx_train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help="Number of traning epochs.")
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help="Number of start epoch.")
    parser.add_argument(
        '--max-frames',
        type=int,
        default=90000,
        help="Maximum number of feature frames in a single batch.") 
    parser.add_argument(
        '--den-scale',
        type=float,
        default=1.0,
        help="denominator scale in mmi loss.")
    parser.add_argument(
        '--xent-scale',
        type=float,
        default=0.1,
        help="Cross entropy loss scale.")
    parser.add_argument(
        '--smooth-scale',
        type=float,
        default=0.1,
        help="Phone distribution smoothing loss scale.")
    return parser


def main():
    args = get_parser().parse_args()

    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    max_frames = args.max_frames
    den_scale = args.den_scale
    xent_scale = args.xent_scale
    smooth_scale = args.smooth_scale

    fix_random_seed(42)

    exp_dir = Path('exp-lstm-adam-mmi-xent-musan-' + 'xent-scale-' + str(xent_scale) + '-smooth-scale-' + str(smooth_scale) )
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

    graph_compiler = MmiTrainingGraphCompiler(
        L_inv=L_inv,
        phones=phone_symbol_table,
        words=word_symbol_table
    )
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)

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
    model = TdnnLstm1c(num_features=40,
                       num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
                       subsampling_factor=3)
    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only
    use_adam = True

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    model.to(device)
    describe(model)

    if use_adam:
        learning_rate = 1e-3
        weight_decay = 5e-4
        optimizer = optim.AdamW(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        # Equivalent to the following in the epoch loop:
        #  if epoch > 6:
        #      curr_learning_rate *= 0.8
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: 1.0 if ep < 7 else 0.8 ** (ep - 6)
        )
    else:
        learning_rate = 5e-5
        weight_decay = 1e-5
        momentum = 0.9
        lr_schedule_gamma = 0.7
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=lr_schedule_gamma,
            last_epoch=start_epoch - 1
        )

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        # LR scheduler can hold multiple learning rates for multiple parameter groups;
        # For now we report just the first LR which we assume concerns most of the parameters.
        curr_learning_rate = lr_scheduler.get_last_lr()[0]
        tb_writer.add_scalar('train/learning_rate', curr_learning_rate, global_batch_idx_train)
        tb_writer.add_scalar('train/epoch', epoch, global_batch_idx_train)

        logging.info('epoch {}, learning rate {}'.format(epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            optimizer=optimizer,
            den_scale=den_scale,
            xent_scale=xent_scale,
            smooth_scale=smooth_scale,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
        )
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
                               objf=objf,
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
        epoch_info_filename = os.path.join(exp_dir, 'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           valid_objf=valid_objf,
                           best_valid_objf=best_valid_objf,
                           best_epoch=best_epoch)

        lr_scheduler.step()

    logging.warning('Done')


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
