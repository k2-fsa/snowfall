#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

import numpy as np
import os
import torch


class CollateFunc(object):
    '''Collate function for LMDataset
    '''

    def __init__(self, pad_index=None):
        # pad_index should be identical to ignore_index of torch.nn.NLLLoss
        # and padding_idx in torch.nn.Embedding
        self.pad_index = pad_index

    def __call__(self, batch: List[List[int]]):
        '''batch contains token_id.
           batch can be viewd as a ragged 2-d array, with a row represents a token_id.
           token_id reprents a tokenized text, whose format is:
           <bos_id> token_id token_id token_id *** <eos_id>
        '''
        # data_pad: [batch_size, seq_len]
        # each seq_len always different
        data_pad = pad_sequence(
            [torch.from_numpy(np.array(x)).long() for x in batch], True,
            self.pad_index)
        data_pad = data_pad.t().contiguous()
        # xs_pad, ys_pad: [max_seq_len, batch_size]
        # max_seq_len is the maximum length in current batch
        xs_pad = data_pad[:-1, :]
        ys_pad = data_pad[1:, :]
        return xs_pad, ys_pad


class LMDataset(Dataset):

    def __init__(self, token_file: str, ntoken: int):
        '''Dataset to load Language Model train/dev text data

        Args:
            token_file: each line is a tokenized text, looks like:
                token_id token_id *** token_id token_id

                A real example is:

                485 135 974 255 1220 33 35 377
                2130 1960

            when loaded, <bos_id>/<eos_id> is added to compose input/target

        '''
        self.bos_id = ntoken - 3
        self.eos_id = ntoken - 2
        self.pad_index = ntoken - 1
        assert os.path.exists(
            token_file
        ), "token_file: {} does not exist, please check that.".format(
            token_file)
        self.data = []
        with open(token_file, 'r') as f:
            for line in f:
                token_id = [int(i) for i in line.strip().split()]
                # Empty line exists in librispeech.txt. Disregrad that.
                if len(token_id) == 0:
                    continue
                # https://github.com/espnet/espnet/blob/master/espnet/lm/lm_utils.py#L179
                # add bos_id and eos_id to each piece of example
                token_id.insert(0, self.bos_id)
                token_id.append(self.eos_id)
                self.data.append(token_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    dev_file = "./data/nnlm/text/dev.txt.tokens"
    dataset = LMDataset(dev_file)
    collate_func = CollateFunc()
    data_loader = DataLoader(dataset,
                             batch_size=2,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=collate_func)
    for i, batch in enumerate(data_loader):
        xs, ys = batch
        print(xs)
        print(ys)
        print(batch)
