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
import torch.nn as nn
import torch.optim as optim
import sys
import yaml

sys.path.insert(0, './local/')

from common import load_checkpoint
from dataset import LMDataset, CollateFunc
from model import TransformerModel
from pathlib import Path
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import List, Dict


def get_args():
    parser = argparse.ArgumentParser(
        description='training Neural Language Model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--vocab_size', type=int, default=3000)
    parser.add_argument('--resume_model_iter',
                        type=int,
                        default=-1,
                        help='resume from trained model;')

    args = parser.parse_args()

    return args


def validate_configs(configs: Dict, required_fields: List) -> bool:
    not_exist_fields = []
    for field in required_fields:
        if field not in configs or configs[field] is None:
            not_exist_fields.append(field)
    if len(not_exist_fields) > 0:
        assert False, 'set following required fields {}'.format(
            ' '.join(not_exist_fields))
    return True


def extract_configs(args) -> Dict:
    assert os.path.exists(args.config), '{} does not exist'.format(args.cofnig)
    required_fields = [
        'model_module', 'shared_conf', 'optimizer_conf', 'trainer_conf',
        'dataset_conf'
    ]
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    validate_configs(configs, required_fields)

    model_conf = '{}_conf'.format(configs['model_module'])
    ntoken = configs['shared_conf']['ntoken']

    configs[model_conf]['ntoken'] = ntoken
    configs['trainer_conf']['ntoken'] = ntoken

    assert 'model_dir' in configs['trainer_conf']
    model_dir = configs['trainer_conf']['model_dir']
    Path(os.path.dirname(model_dir)).mkdir(parents=True, exist_ok=True)

    return configs


def main():
    args = get_args()
    configs = extract_configs(args)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(2021)

    ntoken = args.vocab_size + 3
    assert ntoken == configs['shared_conf']['ntoken']

    # Data
    pad_index = ntoken - 1
    collate_func = CollateFunc(pad_index=pad_index)

    train_dataset = LMDataset(configs['dataset_conf']['train_token'],
                              ntoken=ntoken)
    dev_dataset = LMDataset(configs['dataset_conf']['dev_token'],
                            ntoken=ntoken)

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=collate_func,
                                   **configs['dataloader_conf']['train'])

    dev_data_loader = DataLoader(dev_dataset,
                                 collate_fn=collate_func,
                                 **configs['dataloader_conf']['dev'])

    # initialize or resume model
    if configs['model_module'] == 'transformer':
        model = TransformerModel(**configs['transformer_conf'])

    if args.resume_model_iter > 0:
        model_dir = configs['trainer_conf']['model_dir']
        model_path = '{}/epoch_{}.pt'.format(model_dir, args.resume_model_iter)
        assert os.path.exists(model_path)
        load_checkpoint(model_path, model)

    optimizer = optim.AdamW(model.parameters(), **configs['optimizer_conf'])
    use_cuda = configs['gpu'] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = nn.NLLLoss(ignore_index=pad_index)

    writer = SummaryWriter(log_dir=configs['tensorboard_dir'])

    log_interval = max(100, len(train_data_loader) // 20)
    trainer = Trainer(device=device,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      train_data_loader=train_data_loader,
                      dev_data_loader=dev_data_loader,
                      epoch=args.resume_model_iter + 1,
                      log_interval=log_interval,
                      writer=writer,
                      **configs['trainer_conf'])
    trainer.run()


if __name__ == '__main__':
    main()
