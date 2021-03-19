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
import sys
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet
from lhotse.dataset import BucketingSampler, CutConcatenate, CutMix, K2SpeechRecognitionDataset, SingleCutSampler
from lhotse.utils import fix_random_seed
from snowfall.common import describe
from snowfall.common import load_checkpoint, save_checkpoint, str2bool
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.dist import cleanup_dist, setup_dist
from snowfall.models import AcousticModel
from snowfall.models.tdnn_lstm import TdnnLstm1b
from snowfall.training.diagnostics import measure_gradient_norms, optim_step_and_measure_param_change
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols

den_scale = 1.0



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
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
             optimizer: Optional[torch.optim.Optimizer] = None):
    feature = batch['inputs']
    supervisions = batch['supervisions']
    subsampling_factor = model.module.subsampling_factor if isinstance(model, DDP) else model.subsampling_factor
    supervision_segments = torch.stack(
        (
            supervisions['sequence_idx'],
            torch.floor_divide(supervisions['start_frame'], subsampling_factor),
            torch.floor_divide(supervisions['num_frames'], subsampling_factor)
        ),
        1
    ).to(torch.int32)
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
        nnet_output = model(feature)
    else:
        with torch.no_grad():
            nnet_output = model(feature)

    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

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

    num = k2.intersect_dense(num, dense_fsa_vec, 10.0)
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

    if is_training:
        def maybe_log_gradients(tag: str):
            if (
                    tb_writer is not None
                    and global_batch_idx_train is not None
                    and global_batch_idx_train % 200 == 0
            ):
                tb_writer.add_scalars(
                    tag,
                    measure_gradient_norms(model, norm='l1'),
                    global_step=global_batch_idx_train
                )

        optimizer.zero_grad()
        (-tot_score).backward()
        maybe_log_gradients('train/grad_norms')
        clip_grad_value_(model.parameters(), 5.0)
        maybe_log_gradients('train/clipped_grad_norms')
        if tb_writer is not None and global_batch_idx_train % 200 == 0:
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

    ans = -tot_score.detach().cpu().item(), tot_frames.cpu().item(), all_frames.cpu().item()
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        P: k2.Fsa,
                        device: torch.device,
                        graph_compiler: MmiTrainingGraphCompiler):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        objf, frames, all_frames = get_objf(batch, model, P, device,
                                            graph_compiler, False)
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel, P: k2.Fsa,
                    device: torch.device,
                    graph_compiler: MmiTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    current_epoch: int,
                    tb_writer: Optional[SummaryWriter],
                    num_epochs: int,
                    global_batch_idx_train: int):
    total_objf, total_frames, total_all_frames = 0., 0., 0.
    valid_average_objf = float('inf')
    time_waiting_for_batch = 0
    prev_timestamp = datetime.now()

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()

        if isinstance(model, DDP):
            P.set_scores_stochastic_(model.module.P_scores)
        else:
            P.set_scores_stochastic_(model.P_scores)
        assert P.is_cpu
        assert P.requires_grad is True

        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = get_objf(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
            optimizer=optimizer
        )

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0 and dist.get_rank() == 0:
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

        if batch_idx > 0 and batch_idx % 1000 == 0:
            total_valid_objf, total_valid_frames, total_valid_all_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                P=P,
                device=device,
                graph_compiler=graph_compiler)
            # Synchronize the loss to the master node so that we display it correctly.
            # dist.reduce performs sum reduction by default.
            valid_average_objf = total_valid_objf / total_valid_frames
            model.train()
            if dist.get_rank() == 0:
                logging.info(
                    'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'
                        .format(valid_average_objf,
                                total_valid_frames,
                                100.0 * total_valid_frames / total_valid_all_frames))
            if tb_writer is not None:
                tb_writer.add_scalar('train/global_valid_average_objf',
                                     valid_average_objf,
                                     global_batch_idx_train)
                (model.module if isinstance(model, DDP) else model).write_tensorboard_diagnostics(
                    tb_writer,
                    global_step=global_batch_idx_train
                )
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--bucketing_sampler', type=str2bool, default=True)
    return parser


def main():
    args = get_parser().parse_args()
    print('World size:', args.world_size, 'Rank:', args.local_rank)
    setup_dist(rank=args.local_rank, world_size=args.world_size)
    fix_random_seed(42)

    start_epoch = 0
    num_epochs = 10
    use_adam = True

    exp_dir = f'exp-lstm-adam-mmi-bigram-musan-dist'
    setup_logger('{}/log/log-train'.format(exp_dir), use_console=args.local_rank == 0)
    tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard') if args.local_rank == 0 else None

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

    device_id = args.local_rank
    device = torch.device('cuda', device_id)
    graph_compiler = MmiTrainingGraphCompiler(
        L_inv=L_inv,
        phones=phone_symbol_table,
        words=word_symbol_table,
        device=device
    )
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)
    P = P.to(device)

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
    transforms = [CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20))]
    if not args.bucketing_sampler:
        # We don't mix concatenating the cuts and bucketing
        # Here we insert concatenation before mixing so that the
        # noises from Musan are mixed onto almost-zero-energy
        # padding frames.
        transforms = [CutConcatenate(duration_factor=1)] + transforms
    train = K2SpeechRecognitionDataset(cuts_train, cut_transforms=transforms)
    if args.bucketing_sampler:
        logging.info('Using BucketingSampler.')
        train_sampler = BucketingSampler(
            cuts_train,
            max_frames=40000,
            shuffle=True,
            num_buckets=30
        )
    else:
        logging.info('Using regular sampler with cut concatenation.')
        train_sampler = SingleCutSampler(
            cuts_train,
            max_frames=30000,
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
    # Note: we explicitly set world_size to 1 to disable the auto-detection of
    #       distributed training inside the sampler. This way, every GPU will
    #       perform the computation on the full dev set. It is a bit wasteful,
    #       but unfortunately loss aggregation between multiple processes with
    #       torch.distributed.all_reduce() tends to hang indefinitely inside
    #       NCCL after ~3000 steps. With the current approach, we can still report
    #       the loss on the full validation set.
    valid_sampler = SingleCutSampler(cuts_dev, max_frames=90000, world_size=1, rank=0)
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
    model = TdnnLstm1b(num_features=40,
                       num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
                       subsampling_factor=3)
    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

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
            gamma=lr_schedule_gamma
        )

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer, scheduler=lr_scheduler)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    if args.world_size > 1:
        logging.info('Using DistributedDataParallel in training. '
                     'The reported loss, num_frames, etc. for training steps include '
                     'only the batches seen in the master process (the actual loss '
                     'includes batches from all GPUs, and the actual num_frames is '
                     f'approx. {args.world_size}x larger.')
        # For now do not sync BatchNorm across GPUs due to NCCL hanging in all_gather...
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)

        # LR scheduler can hold multiple learning rates for multiple parameter groups;
        # For now we report just the first LR which we assume concerns most of the parameters.
        curr_learning_rate = lr_scheduler.get_last_lr()[0]
        if tb_writer is not None:
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
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
        )

        lr_scheduler.step()

        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            model=model,
                            optimizer=None,
                            scheduler=None,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            local_rank=args.local_rank,
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
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        local_rank=args.local_rank,
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

    logging.warning('Done')
    cleanup_dist()


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
