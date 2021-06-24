#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Guo Liyong)
# Apache 2.0

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import k2
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lhotse.utils import fix_random_seed, nullcontext
from snowfall.common import describe, str2bool
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.data.librispeech import LibriSpeechAsrDataModule
from snowfall.dist import setup_dist
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.transformer import Noam
from snowfall.text.numericalizer import Numericalizer
from snowfall.training.diagnostics import measure_gradient_norms, optim_step_and_measure_param_change


# DDP seems disable logging as discussed here https://github.com/k2-fsa/snowfall/pull/158
# Temporarily use print instead of logging.info
logging.info = print

def get_objf(batch: Dict,
             model: AcousticModel,
             device: torch.device,
             numericalizer: Numericalizer,
             is_training: bool,
             is_update: bool,
             accum_grad: int = 1,
             att_rate: float = 0.0,
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             ):
    feature = batch['inputs']
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch['supervisions']

    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1
    ).to(torch.int32)

    texts = supervisions['text']

    grad_context = nullcontext if is_training else torch.no_grad
    with grad_context():
        # nnet_output is [N, C, T]
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)

        blank_id = 0
        unk_id = 1 # i.e. oov_id
        token_ids = []
        for text in texts:
            token_ids.append(list(filter(lambda x: x != blank_id and x != unk_id, numericalizer.EncodeAsIds(text.upper()))))


        if att_rate != 0.0:
            att_loss = model.module.decoder_forward(encoder_memory, memory_mask, token_ids=token_ids)

        # Prepare to compute ctc_loss
        nnet_output = nnet_output.permute(2, 0, 1)  # Now is [T, N, C], as required by torch.nn.CTCLoss

        target_lengths = torch.tensor([len(token_id) for token_id in token_ids])  # N
        target= torch.tensor(list(np.concatenate(token_ids).flat))  # size is sum(target_lengths)
        assert sum(target_lengths) == len(target)

        input_lengths = supervision_segments[:,2]

        bni = input_lengths.shape[0]  # batch_size of input
        bno = nnet_output.shape[1]  # batch_size of nnet_output
        bnt = target_lengths.shape[0]  # batch_size of target_legnths
        assert bno == bni and bno == bnt and bni == bnt, 'Inconsistent batch-size!'

        ctc_loss = model.module.ctc_loss_fn(nnet_output, target, input_lengths, target_lengths)

        # Normalized by batch_size
        # Reference: https://github.com/espnet/espnet/blob/master/espnet2/asr/ctc.py#L98
        ctc_loss = ctc_loss.sum() / bno

        if att_rate != 0.0:
            loss = ((1.0 - att_rate) * ctc_loss + att_rate * att_loss) *  accum_grad
        else:
            loss = (ctc_loss) * accum_grad

    if is_training:
        def maybe_log_gradients(tag: str):
            if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
                tb_writer.add_scalars(
                    tag,
                    measure_gradient_norms(model, norm='l1'),
                    global_step=global_batch_idx_train
                )

        loss.backward()
        if is_update:
            maybe_log_gradients('train/grad_norms')

            # Reference: https://github.com/espnet/espnet/blob/a50c37b79da7cd97d86e4f475283c41685468b53/espnet2/train/trainer.py#L585
            clip_grad_norm_(model.parameters(), 5.0, 2.0)
            maybe_log_gradients('train/clipped_grad_norms')
            if tb_writer is not None and (global_batch_idx_train // accum_grad) % 200 == 0:
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
            optimizer.zero_grad()

    # negative loss to get objf
    ans = - loss.detach().cpu().item(), att_loss.cpu().item(), ctc_loss.cpu().item(), bno
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        device: torch.device,
                        numericalizer: Numericalizer,
                        att_rate: int,
                        ):
    model.eval()
    total_loss = 0.0
    total_att_loss = 0.0
    total_ctc_loss = 0.0
    num_utts = 0

    from torchaudio.datasets.utils import bg_iterator
    for batch_idx, batch in enumerate(bg_iterator(dataloader, 2)):
        loss, att_loss, ctc_loss, bsz = get_objf(
            batch=batch,
            model=model,
            device=device,
            numericalizer=numericalizer,
            is_training=False,
            is_update=False,
            att_rate=att_rate,
        )
        total_loss += loss * bsz
        total_att_loss += att_loss * bsz
        total_ctc_loss += ctc_loss * bsz
        num_utts += bsz

    return total_loss, total_att_loss, total_ctc_loss, num_utts


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel,
                    device: torch.device,
                    numericalizer: Numericalizer,
                    optimizer: torch.optim.Optimizer,
                    accum_grad: int,
                    att_rate: float,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int,
                    world_size: int,
                    rank: int,
                    ):
    """One epoch training and validation.

    Args:
        dataloader: Training dataloader
        valid_dataloader: Validation dataloader
        model: Acoustic model to be trained
        device: Training device, torch.device("cpu") or torch.device("cuda", device_id)
        numericalizer: Convert transcript from text --> tokens --> token_ids
        optimizer: Training optimizer
        accum_grad: Number of gradient accumulation
        att_rate: Attention loss rate, final loss is att_rate * att_loss + (1-att_rate) * ctc_loss
        current_epoch: current training epoch, for logging only
        tb_writer: tensorboard SummaryWriter
        num_epochs: total number of training epochs, for logging only
        global_batch_idx_train: global training batch index before this epoch, for logging only
        world_size: indicate multi-gpu training or not
        rank: logging.info will works only rank==0 to avoid duplicate information.

    Returns:
        A tuple of 3 scalar:  (total_objf / epoch_num_utts, valid_average_objf, global_batch_idx_train)
        - `total_objf / total_frames` is the average training loss
        - `valid_average_objf` is the average validation loss
        - `global_batch_idx_train` is the global training batch index after this epoch
    """
    # Return value
    total_objf, epoch_num_utts = 0., 0.

    # To record loginterval batches
    loginterval = 100
    loginterval_loss, loginterval_att_loss, loginterval_ctc_loss, loginterval_num_utts = 0., 0., 0., 0.

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

        curr_batch_objf, curr_batch_att_loss, curr_batch_ctc_loss, curr_batch_num_utts = get_objf(
            batch=batch,
            model=model,
            device=device,
            numericalizer=numericalizer,
            is_training=True,
            is_update=is_update,
            accum_grad=accum_grad,
            att_rate=att_rate,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
            optimizer=optimizer,
        )

        total_objf += curr_batch_objf
        epoch_num_utts += curr_batch_num_utts

        loginterval_loss +=  - curr_batch_objf * curr_batch_num_utts #  objf = - loss
        loginterval_att_loss += curr_batch_att_loss * curr_batch_num_utts
        loginterval_ctc_loss += curr_batch_ctc_loss * curr_batch_num_utts
        loginterval_num_utts += curr_batch_num_utts

        if batch_idx % loginterval == 0 and rank == 0:
            start_batch_idx = max(0, batch_idx - loginterval)  # 0 for start training
            loginterval_loss /= loginterval_num_utts
            loginterval_att_loss /= loginterval_num_utts
            loginterval_ctc_loss /= loginterval_num_utts
            # DDP seems disable logging as discussed here https://github.com/k2-fsa/snowfall/pull/158
            # Temporarily, use print instead of logging.info
            # logging.info(
            #         f'{current_epoch}epoch:train:{start_batch_idx}-{batch_idx}batch: '
            #         f'loss={loginterval_loss:.4}, loss_att={loginterval_att_loss:.4}, loss_ctc={loginterval_ctc_loss:.4}')
            print(
                    f'{current_epoch}epoch:train:{start_batch_idx}-{batch_idx}batch: '
                    f'loss={loginterval_loss:.4}, loss_att={loginterval_att_loss:.4}, loss_ctc={loginterval_ctc_loss:.4}',
                    flush=True)


            if tb_writer is not None:
                tb_writer.add_scalar('train/current_loginterval_loss',
                                     loginterval_loss,
                                     global_batch_idx_train)
                tb_writer.add_scalar('train/current_loginterval_att_loss',
                                     loginterval_att_loss,
                                     global_batch_idx_train)
                tb_writer.add_scalar('train/current_loginreval_ctc_loss',
                                     loginterval_ctc_loss,
                                     global_batch_idx_train)
            loginterval_loss, loginterval_att_loss, loginterval_ctc_loss, loginterval_num_utts = 0., 0., 0., 0.
            # if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 1000 == 0:
            total_valid_objf, _, _, valid_num_utts = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
                numericalizer=numericalizer,
                att_rate=att_rate)

            if world_size > 1:
                s = torch.tensor([
                    total_valid_objf,
                    valid_num_utts,
                ]).to(device)

                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                total_valid_objf, valid_num_utts = s.cpu().tolist()
            valid_average_objf = total_valid_objf / valid_num_utts

            model.train()
            if rank == 0:
                # DDP seems disable logging as discussed here https://github.com/k2-fsa/snowfall/pull/158
                # Temporarily, use print instead of logging.info
                # logging.info(
                #     'Validation average objf: {:.6f} over {} utts'
                #         .format(valid_average_objf,
                #                 valid_num_utts))
                print('Validation average objf: {:.6f} over {} utts'
                        .format(valid_average_objf,
                                valid_num_utts), flush=True)

            if tb_writer is not None:
                tb_writer.add_scalar('train/global_valid_average_objf',
                                     valid_average_objf,
                                     global_batch_idx_train)
                model.module.write_tensorboard_diagnostics(tb_writer, global_step=global_batch_idx_train)
        prev_timestamp = datetime.now()
    return total_objf / epoch_num_utts, valid_average_objf, global_batch_idx_train


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--world-size',
        type=int,
        default=1,
        help='Number of GPUs for DDP training.')
    parser.add_argument(
        '--master-port',
        type=int,
        default=12354,
        help='Master port to use for DDP training.')
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer", "contextnet"],
        help="Model type.")
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help="Number of training epochs.")
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help="Number of start epoch.")
    parser.add_argument(
        '--warm-step',
        type=int,
        default=40000,
        help='The number of warm-up steps for Noam optimizer.'
    )
    parser.add_argument(
        '--lr-factor',
        type=float,
        default=10.0,
        help='Learning rate factor for Noam optimizer.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='weight decay (L2 penalty) for Noam optimizer.'
    )
    parser.add_argument(
        '--accum-grad',
        type=int,
        default=1,
        help="Number of gradient accumulation.")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.7,
        help="Attention loss rate.")
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
        '--tensorboard',
        type=str2bool,
        default=True,
        help='Should various information be logged in tensorboard.'
    )
    return parser


def run(rank, world_size, args):
    '''
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    '''
    assert args.start_epoch >= 0
    model_type = args.model_type
    curr_epoch = args.start_epoch + 1
    num_epochs = args.num_epochs
    accum_grad = args.accum_grad
    att_rate = args.att_rate
    attention_dim = args.attention_dim
    nhead=args.nhead

    fix_random_seed(42)
    setup_dist(rank, world_size, args.master_port)

    exp_dir = Path(f'exp-bpe-lrfactor{args.lr_factor}-{model_type}-{attention_dim}-{nhead}-noam')
    setup_logger(f'{exp_dir}/log/log-train-{rank}')
    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')
    else:
        tb_writer = None

    device_id = rank
    device = torch.device('cuda', device_id)


    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    if not torch.cuda.is_available():
        logging.error('no gpu detected!')
        sys.exit(-1)

    # TODO(Liyong Guo) make this configurable.
    lang_dir = Path('data/en_token_list/bpe_unigram5000/')
    bpe_model_path = lang_dir / 'bpe.model'
    tokens_file = lang_dir / 'tokens.txt'
    numericalizer = Numericalizer.build_numericalizer(bpe_model_path, tokens_file)

    if rank == 0:
        logging.info("about to create model")

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
        raise NotImplementedError("model of type " + str(model_type) + " is not verified")


    model.to(device)
    if rank == 0:
        describe(model)

    model = DDP(model, device_ids=[rank])


    optimizer = Noam(model.parameters(),
                     model_size=args.attention_dim,
                     factor=args.lr_factor,
                     warm_step=args.warm_step,
                     weight_decay=args.weight_decay)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = curr_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if args.start_epoch > 1:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    # The first trained model is named as: epoch-1.pt not epoch-0.pt.
    # curr_epoch is belong to [1, num_epochs]
    for epoch in range(curr_epoch, num_epochs + 1):
        train_dl.sampler.set_epoch(epoch)
        curr_learning_rate = optimizer._rate
        if tb_writer is not None:
            tb_writer.add_scalar('train/learning_rate', curr_learning_rate, global_batch_idx_train)
            tb_writer.add_scalar('train/epoch', epoch, global_batch_idx_train)

        if rank == 0:
            logging.info('epoch {}, learning rate {}'.format(epoch, curr_learning_rate))

        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            device=device,
            numericalizer=numericalizer,
            optimizer=optimizer,
            accum_grad=accum_grad,
            att_rate=att_rate,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
            world_size=world_size,
            rank=rank,
        )
        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            optimizer=None,
                            scheduler=None,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            valid_objf=valid_objf,
                            global_batch_idx_train=global_batch_idx_train,
                            local_rank=rank)
            save_training_info(filename=best_epoch_info_filename,
                               model_path=best_model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               valid_objf=valid_objf,
                               best_valid_objf=best_valid_objf,
                               best_epoch=best_epoch,
                               local_rank=rank)

        # we always save the model for every epoch
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        optimizer=optimizer,
                        scheduler=None,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        valid_objf=valid_objf,
                        global_batch_idx_train=global_batch_idx_train,
                        local_rank=rank)
        epoch_info_filename = os.path.join(exp_dir, 'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           valid_objf=valid_objf,
                           best_valid_objf=best_valid_objf,
                           best_epoch=best_epoch,
                           local_rank=rank)

    logging.warning('Done')
    torch.distributed.barrier()
    cleanup_dist()


def main():
    parser = get_parser()
    print('config parsed')
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    world_size = args.world_size
    assert world_size >= 1
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
