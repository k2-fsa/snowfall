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
        # max_seq_len is the maximum lenght in current batch
        xs_pad = data_pad[:-1, :]
        ys_pad = data_pad[1:, :]
        return xs_pad, ys_pad


class LMDataset(Dataset):

    def __init__(self, text_file: str, ntokens=None):
        '''Dataset to load Language Model train/dev text data

        Args:
            text_file: text file, text for one utt per line.
        '''
        self.bos_id = ntokens - 3
        self.eos_id = ntokens - 2
        self.pad_index = ntokens - 1
        assert os.path.exists(
            text_file
        ), "text_file: {} does not exist, please check that.".format(text_file)
        self.data = []
        with open(text_file, 'r') as f:
            for idx, line in enumerate(f):
                token_id = [int(i) for i in line.strip().split()]
                # add bos_id and eos_id to each piece of example
                # then each valid example should be longer than 2
                token_id.insert(0, self.bos_id)
                token_id.append(self.eos_id)
                if len(token_id) > 2:
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
