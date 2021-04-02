#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

# Reference:
# https://github.com/mobvoi/wenet/blob/main/wenet/bin/train.py
import argparse

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.insert(0, './local/')

from common import load_checkpoint
from dataset import LMDataset, CollateFunc
from model import TransformerModel, RNNModel
from pathlib import Path
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(
        description='training Neural Language Model')
    parser.add_argument('--train_token',
                        default='data/nnlm/text/librispeech.txt.tokens',
                        help='train data file')
    parser.add_argument('--dev_token',
                        default='data/nnlm/text/dev.txt.tokens',
                        help='dev data file')
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--emsize', type=int, default=200)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--nhid', type=int, default=200)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-2,
                        help='initial learning rate')
    parser.add_argument('--clip',
                        type=float,
                        default=50.0,
                        help='gradient clipping')
    parser.add_argument('--model_dir',
                        default='./exp-nnlm/models/',
                        help='path to save model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='path to save tensorboard log')
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument(
        '--model_iter',
        type=int,
        default=-1,
        help='resume from trained model; if -1 training from scratch')
    parser.add_argument('--model_type',
                        type=str,
                        default='Transformer',
                        help='model type')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(2021)
    # args.vocab_size: number of tokens in tokenizer.get_vocab
    # + 2: one for eos_id, another for pad_idx
    # i.e. token_idxs[0, 1, 2, ...., ntokens -3, ntokens - 2, ntokens - 1]
    # bos_id: ntokens - 3
    # eos_id: ntokens - 2
    # pad_idx: ntokens - 1
    ntokens = args.vocab_size + 3
    pad_index = ntokens - 1

    collate_func = CollateFunc(pad_index=pad_index)

    train_dataset = LMDataset(args.train_token, ntokens=ntokens)
    dev_dataset = LMDataset(args.dev_token, ntokens=ntokens)

    # To debug dataset.py, set shuffle=False and num_workers=0
    # then examples will be loaded as the sequence they are in {train, dev}.tokens
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True,
                                   collate_fn=collate_func)

    dev_data_loader = DataLoader(dev_dataset,
                                 batch_size=20,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True,
                                 collate_fn=collate_func)

    if 'Trasformer' == args.model_type:
        model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                                 args.nlayers, args.dropout)
    else:
        model = RNNModel('LSTM', ntokens, args.emsize, args.nhid, args.nlayers,
                         args.dropout, False)

    if args.model_iter > 0:
        model_path = '{}/epoch_{}.pt'.format(args.model_dir, args.model_iter)
        load_checkpoint(model_path, model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)
    criterion = nn.NLLLoss(ignore_index=pad_index)
    exp_dir = 'exp-nnlm'
    writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')

    Path(os.path.dirname(args.model_dir)).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(device,
                      model,
                      criterion,
                      optimizer,
                      train_data_loader=train_data_loader,
                      dev_data_loader=dev_data_loader,
                      ntokens=ntokens,
                      batch_size=args.batch_size,
                      epoch=args.model_iter + 1,
                      num_epochs=args.num_epochs,
                      clip=args.clip,
                      model_dir=args.model_dir,
                      writer=writer)
    trainer.run()


if __name__ == '__main__':
    main()
