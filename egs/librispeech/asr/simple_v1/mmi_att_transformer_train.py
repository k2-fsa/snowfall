#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
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
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple

from lhotse import CutSet, Fbank, load_manifest
from lhotse.dataset import BucketingSampler, CutConcatenate, CutMix, K2SpeechRecognitionDataset, SingleCutSampler, \
    SpecAugment
from lhotse.dataset.cut_transforms.perturb_speed import PerturbSpeed
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from snowfall.common import describe, str2bool
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.transformer import Noam, Transformer
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
             is_update: bool,
             accum_grad: int = 1,
             den_scale: float = 1.0,
             att_rate: float = 0.0,
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
             optimizer: Optional[torch.optim.Optimizer] = None):
    feature = batch['inputs']
    supervisions = batch['supervisions']
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
    supervision_segments = torch.clamp(supervision_segments, min=0)
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
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
        if att_rate != 0.0:
            att_loss = model.decoder_forward(encoder_memory, memory_mask, supervisions, graph_compiler)
    else:
        with torch.no_grad():
            nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
            if att_rate != 0.0:
                att_loss = model.decoder_forward(encoder_memory, memory_mask, supervisions, graph_compiler)

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
            if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
                tb_writer.add_scalars(
                    tag,
                    measure_gradient_norms(model, norm='l1'),
                    global_step=global_batch_idx_train
                )

        if att_rate != 0.0:
            loss = (- (1.0 - att_rate) * tot_score + att_rate * att_loss) / (len(texts) * accum_grad)
        else:
            loss = (-tot_score) / (len(texts) * accum_grad)
        loss.backward()
        if is_update:
            maybe_log_gradients('train/grad_norms')
            clip_grad_value_(model.parameters(), 5.0)
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

    ans = -tot_score.detach().cpu().item(), tot_frames.cpu().item(
    ), all_frames.cpu().item()
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        P: k2.Fsa,
                        device: torch.device,
                        graph_compiler: MmiTrainingGraphCompiler,
                        den_scale: float = 1
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
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=False,
            is_update=False,
            den_scale=den_scale
        )
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
                    accum_grad: int,
                    den_scale: float,
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

        if forward_count == 1 or accum_grad == 1:
            P.set_scores_stochastic_(model.P_scores)
            assert P.requires_grad is True

        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = get_objf(
            batch=batch,
            model=model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            is_update=is_update,
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
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
                P=P,
                device=device,
                graph_compiler=graph_compiler)
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
                model.write_tensorboard_diagnostics(tb_writer, global_step=global_batch_idx_train)
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer"],
        help="Model type.")
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
        '--max-duration',
        type=int,
        default=500.0,
        help="Maximum pooled recordings duration (seconds) in a single batch.")
    parser.add_argument(
        '--warm-step',
        type=int,
        default=5000,
        help='The number of warm-up steps for Noam optimizer.'
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
        '--bucketing_sampler',
        type=str2bool,
        default=False,
        help='When enabled, the batches will come from buckets of '
             'similar duration (saves padding frames).')
    parser.add_argument(
        '--num-buckets',
        type=int,
        default=30,
        help='The number of buckets for the BucketingSampler'
             '(you might want to increase it for larger datasets).')
    parser.add_argument(
        '--concatenate-cuts',
        type=str2bool,
        default=True,
        help='When enabled, utterances (cuts) will be concatenated '
             'to minimize the amount of padding.')
    parser.add_argument(
        '--duration-factor',
        type=float,
        default=1.0,
        help='Determines the maximum duration of a concatenated cut '
             'relative to the duration of the longest cut in a batch.')
    parser.add_argument(
        '--gap',
        type=float,
        default=1.0,
        help='The amount of padding (in seconds) inserted between concatenated cuts. '
             'This padding is filled with noise when noise augmentation is used.')
    parser.add_argument(
        '--full-libri',
        type=str2bool,
        default=False,
        help='When enabled, use 960h LibriSpeech.')
    parser.add_argument(
        '--on-the-fly-feats',
        type=str2bool,
        default=False,
        help='When enabled, use on-the-fly cut mixing and feature extraction. '
             'Will drop existing precomputed feature manifests if available.'
    )
    parser.add_argument(
        '--tensorboard',
        type=str2bool,
        default=True,
        help='Should various information be logged in tensorboard.'
    )
    return parser


def main():
    args = get_parser().parse_args()

    model_type = args.model_type
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    max_duration = args.max_duration
    accum_grad = args.accum_grad
    den_scale = args.den_scale
    att_rate = args.att_rate

    fix_random_seed(42)

    exp_dir = Path('exp-' + model_type + '-noam-mmi-att-musan-sa')
    setup_logger('{}/log/log-train'.format(exp_dir))
    tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard') if args.tensorboard else None

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

    device_id = 0
    device = torch.device('cuda', device_id)

    graph_compiler = MmiTrainingGraphCompiler(
        L_inv=L_inv,
        phones=phone_symbol_table,
        words=word_symbol_table,
        device=device,
    )
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)
    P = P.to(device)

    # load dataset
    feature_dir = Path('exp/data')
    logging.info("About to get train cuts")
    cuts_train = load_manifest(feature_dir / 'cuts_train-clean-100.json.gz')
    if args.full_libri:
        cuts_train = (
            cuts_train +
            load_manifest(feature_dir / 'cuts_train-clean-360.json.gz') +
            load_manifest(feature_dir / 'cuts_train-other-500.json.gz')
        )
    logging.info("About to get dev cuts")
    cuts_dev = (
        load_manifest(feature_dir / 'cuts_dev-clean.json.gz') +
        load_manifest(feature_dir / 'cuts_dev-other.json.gz')
    )
    logging.info("About to get Musan cuts")
    cuts_musan = load_manifest(feature_dir / 'cuts_musan.json.gz')

    logging.info("About to create train dataset")
    transforms = [CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20))]
    if args.concatenate_cuts:
        logging.info(f'Using cut concatenation with duration factor {args.duration_factor} and gap {args.gap}.')
        # Cut concatenation should be the first transform in the list,
        # so that if we e.g. mix noise in, it will fill the gaps between different utterances.
        transforms = [CutConcatenate(duration_factor=args.duration_factor, gap=args.gap)] + transforms
    train = K2SpeechRecognitionDataset(
        cuts_train,
        cut_transforms=transforms,
        input_transforms=[
             SpecAugment(num_frame_masks=2, features_mask_size=27, num_feature_masks=2, frames_mask_size=100)
        ]
    )

    if args.on_the_fly_feats:
        # NOTE: the PerturbSpeed transform should be added only if we remove it from data prep stage.
        # # Add on-the-fly speed perturbation; since originally it would have increased epoch
        # # size by 3, we will apply prob 2/3 and use 3x more epochs.
        # # Speed perturbation probably should come first before concatenation,
        # # but in principle the transforms order doesn't have to be strict (e.g. could be randomized)
        # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)] + transforms
        # Drop feats to be on the safe side.
        cuts_train = cuts_train.drop_features()
        from lhotse.features.fbank import FbankConfig
        train = K2SpeechRecognitionDataset(
            cuts=cuts_train,
            cut_transforms=transforms,
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            input_transforms=[
                SpecAugment(num_frame_masks=2, features_mask_size=27, num_feature_masks=2, frames_mask_size=100)
            ]
        )

    if args.bucketing_sampler:
        logging.info('Using BucketingSampler.')
        train_sampler = BucketingSampler(
            cuts_train,
            max_duration=max_duration,
            shuffle=True,
            num_buckets=args.num_buckets
        )
    else:
        logging.info('Using SingleCutSampler.')
        train_sampler = SingleCutSampler(
            cuts_train,
            max_duration=max_duration,
            shuffle=True,
        )
    logging.info("About to create train dataloader")
    train_dl = torch.utils.data.DataLoader(
        train,
        sampler=train_sampler,
        batch_size=None,
        num_workers=4,
    )

    logging.info("About to create dev dataset")
    if args.on_the_fly_feats:
        cuts_dev = cuts_dev.drop_features()
        validate = K2SpeechRecognitionDataset(
            cuts_dev.drop_features(),
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
        )
    else:
        validate = K2SpeechRecognitionDataset(cuts_dev)
    valid_sampler = SingleCutSampler(
        cuts_dev,
        max_duration=max_duration,
    )
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
            num_decoder_layers=num_decoder_layers)
    else:
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers)

    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

    model.to(device)
    describe(model)

    optimizer = Noam(model.parameters(),
                     model_size=args.attention_dim,
                     factor=1.0,
                     warm_step=args.warm_step)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer)
        best_objf = ckpt['objf']
        best_valid_objf = ckpt['valid_objf']
        global_batch_idx_train = ckpt['global_batch_idx_train']
        logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        curr_learning_rate = optimizer._rate
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
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
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
                            optimizer=None,
                            scheduler=None,
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
                        optimizer=optimizer,
                        scheduler=None,
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

    logging.warning('Done')


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
