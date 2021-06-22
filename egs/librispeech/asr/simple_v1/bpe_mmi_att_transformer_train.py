#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import k2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lhotse.utils import fix_random_seed, nullcontext
from snowfall.common import describe, str2bool
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.data.librispeech import LibriSpeechAsrDataModule
from snowfall.dist import cleanup_dist
from snowfall.dist import setup_dist
from snowfall.lexicon import Lexicon
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.contextnet import ContextNet
from snowfall.models.tdnn_lstm import TdnnLstm1b  # alignment model
from snowfall.models.transformer import Noam, Transformer
from snowfall.objectives import LFMMILoss, encode_supervisions
from snowfall.training.diagnostics import measure_gradient_norms, optim_step_and_measure_param_change
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler


def get_objf(batch: Dict,
             model: AcousticModel,
             ali_model: Optional[AcousticModel],
             device: torch.device,
             graph_compiler: MmiTrainingGraphCompiler,
             use_pruned_intersect: bool,
             is_training: bool,
             is_update: bool,
             accum_grad: int = 1,
             den_scale: float = 1.0,
             att_rate: float = 0.0,
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scaler: GradScaler = None
             ):
    feature = batch['inputs']
    # at entry, feature is [N, T, C]
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch['supervisions']
    supervision_segments, texts = encode_supervisions(supervisions)

    loss_fn = LFMMILoss(
        graph_compiler=graph_compiler,
        den_scale=den_scale,
        use_pruned_intersect=use_pruned_intersect
    )

    grad_context = nullcontext if is_training else torch.no_grad

    with autocast(enabled=scaler.is_enabled()), grad_context():

        if att_rate == 0:
            # Note: Make TorchScript happy by making the supervision dict strictly
            #       conform to type Dict[str, Tensor]
            #       Using the attention decoder with TorchScript is currently unsupported,
            #       we'll need to separate out the 'text' field from 'supervisions' first.
            del supervisions['text']

        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        if att_rate != 0.0:
            att_loss = model.module.decoder_forward(encoder_memory, memory_mask, supervisions, graph_compiler)

        if (ali_model is not None and global_batch_idx_train is not None and
                global_batch_idx_train // accum_grad < 4000):
            with torch.no_grad():
                ali_model_output = ali_model(feature)
            # subsampling is done slightly differently, may be small length
            # differences.
            min_len = min(ali_model_output.shape[2], nnet_output.shape[2])
            # scale less than one so it will be encouraged
            # to mimic ali_model's output
            ali_model_scale = 500.0 / (global_batch_idx_train // accum_grad + 500)
            nnet_output = nnet_output.clone()  # or log-softmax backprop will fail.
            nnet_output[:, :,:min_len] += ali_model_scale * ali_model_output[:, :,:min_len]

        # nnet_output is [N, C, T]
        nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]

        mmi_loss, tot_frames, all_frames = loss_fn(nnet_output, texts, supervision_segments)

    if is_training:
        def maybe_log_gradients(tag: str):
            if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
                tb_writer.add_scalars(
                    tag,
                    measure_gradient_norms(model, norm='l1'),
                    global_step=global_batch_idx_train
                )

        if att_rate != 0.0:
            loss = (- (1.0 - att_rate) * mmi_loss + att_rate * att_loss) / (len(texts) * accum_grad)
        else:
            loss = (-mmi_loss) / (len(texts) * accum_grad)
        scaler.scale(loss).backward()
        if is_update:
            maybe_log_gradients('train/grad_norms')
            scaler.unscale_(optimizer)
            clip_grad_value_(model.parameters(), 5.0)
            maybe_log_gradients('train/clipped_grad_norms')
            if tb_writer is not None and (global_batch_idx_train // accum_grad) % 200 == 0:
                # Once in a time we will perform a more costly diagnostic
                # to check the relative parameter change per minibatch.
                deltas = optim_step_and_measure_param_change(model, optimizer, scaler)
                tb_writer.add_scalars(
                    'train/relative_param_change_per_minibatch',
                    deltas,
                    global_step=global_batch_idx_train
                )
            else:
                scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

    ans = -mmi_loss.detach().cpu().item(), tot_frames.cpu().item(
    ), all_frames.cpu().item()
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        ali_model: Optional[AcousticModel],
                        device: torch.device,
                        graph_compiler: MmiTrainingGraphCompiler,
                        use_pruned_intersect: bool,
                        scaler: GradScaler,
                        den_scale: float = 1,
                        ):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    from torchaudio.datasets.utils import bg_iterator
    for batch_idx, batch in enumerate(bg_iterator(dataloader, 2)):
        objf, frames, all_frames = get_objf(
            batch=batch,
            model=model,
            ali_model=ali_model,
            device=device,
            graph_compiler=graph_compiler,
            use_pruned_intersect=use_pruned_intersect,
            is_training=False,
            is_update=False,
            den_scale=den_scale,
            scaler=scaler
        )
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel,
                    ali_model: Optional[AcousticModel],
                    device: torch.device,
                    graph_compiler: MmiTrainingGraphCompiler,
                    use_pruned_intersect: bool,
                    optimizer: torch.optim.Optimizer,
                    accum_grad: int,
                    den_scale: float,
                    att_rate: float,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int,
                    world_size: int,
                    scaler: GradScaler
                    ):
    """One epoch training and validation.

    Args:
        dataloader: Training dataloader
        valid_dataloader: Validation dataloader
        model: Acoustic model to be trained
        P: An FSA representing the bigram phone LM
        device: Training device, torch.device("cpu") or torch.device("cuda", device_id)
        graph_compiler: MMI training graph compiler
        optimizer: Training optimizer
        accum_grad: Number of gradient accumulation
        den_scale: Denominator scale in mmi loss
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
            ali_model=ali_model,
            device=device,
            graph_compiler=graph_compiler,
            use_pruned_intersect=use_pruned_intersect,
            is_training=True,
            is_update=is_update,
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
            optimizer=optimizer,
            scaler=scaler
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

            if tb_writer is not None:
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
                ali_model=ali_model,
                device=device,
                graph_compiler=graph_compiler,
                use_pruned_intersect=use_pruned_intersect,
                scaler=scaler)
            if world_size > 1:
                s = torch.tensor([
                    total_valid_objf, total_valid_frames,
                    total_valid_all_frames
                ]).to(device)

                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                total_valid_objf, total_valid_frames, total_valid_all_frames = s.cpu().tolist()

            valid_average_objf = total_valid_objf / total_valid_frames
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'
                    .format(valid_average_objf,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            if tb_writer is not None:
                tb_writer.add_scalar('train/global_valid_average_objf',
                                     valid_average_objf,
                                     global_batch_idx_train)
                model.module.write_tensorboard_diagnostics(tb_writer, global_step=global_batch_idx_train)
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


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
        default=5000,
        help='The number of warm-up steps for Noam optimizer.'
    )
    parser.add_argument(
        '--lr-factor',
        type=float,
        default=1.0,
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
        '--den-scale',
        type=float,
        default=1.0,
        help="denominator scale in mmi loss.")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=4,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=256,
        help="Number of units in transformer attention layers.")
    parser.add_argument(
        '--tensorboard',
        type=str2bool,
        default=True,
        help='Should various information be logged in tensorboard.'
    )
    parser.add_argument(
        '--amp',
        type=str2bool,
        default=True,
        help='Should we use automatic mixed precision (AMP) training.'
    )
    parser.add_argument(
        '--use-ali-model',
        type=str2bool,
        default=True,
        help='If true, we assume that you have run ./ctc_train.py '
             'and you have some checkpoints inside the directory '
             'exp-lstm-adam-ctc-musan/ .'
             'It will use exp-lstm-adam-ctc-musan/epoch-{ali-model-epoch}.pt '
             'as the pre-trained alignment model'
    )
    parser.add_argument(
        '--ali-model-epoch',
        type=int,
        default=7,
        help='If --use-ali-model is True, load '
             'exp-lstm-adam-ctc-musan/epoch-{ali-model-epoch}.pt as the alignment model.'
             'Used only if --use-ali-model is True.'
    )
    parser.add_argument(
        '--use-pruned-intersect',
        type=str2bool,
        default=False,
        help='True to use pruned intersect to compute the denominator lattice. ' \
             'You probably want to set it to True if you have a very large LM. ' \
             'In that case, you will get an OOM if it is False. ')
    #  See https://github.com/k2-fsa/k2/issues/739 for more details
    parser.add_argument(
        '--torchscript',
        type=str2bool,
        default=False,
        help='Should we convert the model to TorchScript before starting training.'
    )
    parser.add_argument(
        '--torchscript-epoch',
        type=int,
        default=-1,
        help='After which epoch should we start storing models with TorchScript,'
             'so that they can be simply loaded with torch.jit.load(). '
             '-1 disables this option.'
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
    model_type = args.model_type
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    accum_grad = args.accum_grad
    den_scale = args.den_scale
    att_rate = args.att_rate
    use_pruned_intersect = args.use_pruned_intersect

    fix_random_seed(42)
    setup_dist(rank, world_size, args.master_port)

    exp_dir = Path('exp-bpe-' + model_type + '-mmi-att-sa-vgg-normlayer')
    setup_logger(f'{exp_dir}/log/log-train-{rank}')
    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')
    else:
        tb_writer = None
    #  tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard') if args.tensorboard and rank == 0 else None

    logging.info("Loading lexicon and symbol tables")
    lang_dir = Path('data/lang_bpe2')
    lexicon = Lexicon(lang_dir)

    device_id = rank
    device = torch.device('cuda', device_id)

    graph_compiler = MmiTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )
    phone_ids = lexicon.phone_symbols()

    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    if use_pruned_intersect:
        logging.info('Use pruned intersect for den_lats')
    else:
        logging.info("Don't use pruned intersect for den_lats")

    logging.info("About to create model")

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    if model_type == "transformer":
        model = Transformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=True)
    elif model_type == "conformer":
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=True,
            is_espnet_structure=True)
    elif model_type == "contextnet":
        model = ContextNet(
            num_features=80,
            num_classes=len(phone_ids) + 1)  # +1 for the blank symbol
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not implemented")

    if args.torchscript:
        logging.info('Applying TorchScript to model...')
        model = torch.jit.script(model)

    model.to(device)
    describe(model)

    model = DDP(model, device_ids=[rank])

    # Now for the alignment model, if any
    if args.use_ali_model:
        ali_model = TdnnLstm1b(
            num_features=80,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4)

        ali_model_fname = Path(f'exp-lstm-adam-ctc-musan/epoch-{args.ali_model_epoch}.pt')
        assert ali_model_fname.is_file(), \
                f'ali model filename {ali_model_fname} does not exist!'
        ali_model.load_state_dict(torch.load(ali_model_fname, map_location='cpu')['state_dict'])
        ali_model.to(device)

        ali_model.eval()
        ali_model.requires_grad_(False)
        logging.info(f'Use ali_model: {ali_model_fname}')
    else:
        ali_model = None
        logging.info('No ali_model')

    optimizer = Noam(model.parameters(),
                     model_size=args.attention_dim,
                     factor=args.lr_factor,
                     warm_step=args.warm_step,
                     weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=args.amp)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer, scaler=scaler)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    for epoch in range(start_epoch, num_epochs):
        train_dl.sampler.set_epoch(epoch)
        curr_learning_rate = optimizer._rate
        if tb_writer is not None:
            tb_writer.add_scalar('train/learning_rate', curr_learning_rate, global_batch_idx_train)
            tb_writer.add_scalar('train/epoch', epoch, global_batch_idx_train)

        logging.info('epoch {}, learning rate {}'.format(epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            ali_model=ali_model,
            device=device,
            graph_compiler=graph_compiler,
            use_pruned_intersect=use_pruned_intersect,
            optimizer=optimizer,
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
            world_size=world_size,
            scaler=scaler
        )
        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            optimizer=None,
                            scheduler=None,
                            scaler=None,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            valid_objf=valid_objf,
                            global_batch_idx_train=global_batch_idx_train,
                            local_rank=rank,
                            torchscript=args.torchscript_epoch != -1 and epoch >= args.torchscript_epoch
                            )
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
                        scaler=scaler,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        valid_objf=valid_objf,
                        global_batch_idx_train=global_batch_idx_train,
                        local_rank=rank,
                        torchscript=args.torchscript_epoch != -1 and epoch >= args.torchscript_epoch
                        )
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
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    world_size = args.world_size
    assert world_size >= 1
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
