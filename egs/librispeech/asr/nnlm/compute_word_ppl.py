#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

# Reference:
# https://github.com/espnet/espnet/blob/master/espnet/lm/pytorch_backend/lm.py
# https://github.com/mobvoi/wenet/blob/main/wenet/bin/train.py
import argparse

import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import sys
import yaml

sys.path.insert(0, './local/')

from common import load_checkpoint
from evaluator import Evaluator
# from model import TransformerModel
from pathlib import Path
from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser(
        description='compute token/word ppl of txt')
    parser.add_argument('--config',
                        help='config file',
                        default='conf/lm_small_transformer.yaml')
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--model',
                        type=str,
                        default='exp-nnlm/models/epoch_30.pt',
                        help='full path of loaded model')
    parser.add_argument('--tokenizer_path',
                        type=str,
                        default='exp-nnlm/tokenizer-librispeech.json')
    parser.add_argument('--txt_file',
                        type=str,
                        default='data/nnlm/text/dev.txt')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(2021)

    # device = torch.device("cuda", args.local_rank)
    device = torch.device('cpu')
    print(device)

    evaluator = Evaluator(device=device,
                          model_path=args.model,
                          config_file=args.config,
                          tokenizer_path=args.tokenizer_path)
    evaluator.compute_ppl(txt_file=args.txt_file)


if __name__ == '__main__':
    main()
