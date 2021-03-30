#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

# Reference:
# https://github.com/mobvoi/wenet/blob/main/wenet/bin/train.py
import argparse

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.insert(0, './local/')
sys.path.insert(0, './scripts/')
from lexicon import Lexicon

from dataset import LMDataset, CollateFunc
from model import TransformerModel
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(
        description='training Neural Language Model')
    parser.add_argument('--train_text',
                        default='data/nnlm/text/librispeech.txt',
                        help='train data file')
    parser.add_argument('--dev_text',
                        default='data/nnlm/text/dev.txt',
                        help='dev data file')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ntokens', type=int, default=10000)
    parser.add_argument('--emsize', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=6)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--model_dir',
                        default='./exp/',
                        help='path to save model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='path to save tensorboard log')
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--lexicon-path',
                        default='data/nnlm/lexicon',
                        type=str,
                        help="path to save lexicon files")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    #Set random seed
    torch.manual_seed(2021)
    collate_func = CollateFunc()
    lexicon_filename = '{}/lexicon.txt'.format(args.lexicon_path)
    word2id_filename = '{}/words.txt'.format(args.lexicon_path)
    piece2id_filename = '{}/tokens.txt'.format(args.lexicon_path)

    lexicon = Lexicon(lexicon_filename, word2id_filename, piece2id_filename)
    train_dataset = LMDataset(args.train_text, lexicon)
    dev_dataset = LMDataset(args.dev_text, lexicon)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_func)

    dev_data_loader = DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_func)

    ntokens = args.ntokens
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid,
                             args.nlayers, args.dropout)
    optimizer = optim.Adam(model.parameters())
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = nn.NLLLoss(ignore_index=0)
    exp_dir = 'exp-nnlm'
    writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')
    trainer = Trainer(device,
                      model,
                      criterion,
                      optimizer,
                      train_data_loader=train_data_loader,
                      dev_data_loader=dev_data_loader,
                      ntokens=ntokens,
                      batch_size=args.batch_size,
                      epoch=0,
                      writer=writer)
    trainer.run()


if __name__ == '__main__':
    main()
