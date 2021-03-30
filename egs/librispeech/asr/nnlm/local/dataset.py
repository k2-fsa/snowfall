#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List

import numpy as np
import os
import torch


class CollateFunc(object):
    '''Collate function for LMDataset
    '''

    def __init__(self, pad_index=0):
        # pad_index should be identical to ignore_index of torch.nn.NLLLoss
        self.pad_index = pad_index

    def __call__(self, batch: List[List[int]]):
        '''batch contains token_id.
           batch can be viewd as a ragged 2-d array, with a row represent a token_id.
           token_id reprents a tokenized text, whose format is:
           <bos_id> token_id token_id token_id *** <eos_id>
        '''
        data_pad = pad_sequence(
            [torch.from_numpy(np.array(x)).float() for x in batch], True,
            self.pad_index)
        xs_pad = data_pad[:, :-1]
        ys_pad = data_pad[:, 1:]
        return xs_pad, ys_pad


class LMDataset(Dataset):

    def __init__(self, text_file: str):
        '''Dataset to load Language Model train/dev text data

        Args:
            text_file: text file, text for one utt per line.
        '''
        assert os.path.exists(
            text_file), "text_file: {} does not exist, please check that."
        self.data = []
        with open(text_file, 'r') as f:
            # a line represent a piece of text, e.g.
            # DELAWARE IS NOT AFRAID OF DOGS
            for line in f:
                text = line.strip().split()
                assert len(text) > 0
                text_id = self.text2id(text)
                # token_id format:
                # <bos_id> token_id token_id token_id *** <eos_id>
                token_id = self.text_id2token_id(text_id)
                self.data.append(token_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def text2id(self, text: List[str]) -> List[int]:
        # A dumpy implementation
        return [i for i in range(len(text))]

    def text_id2token_id(self, text_id: List[int]) -> List[int]:
        # A dumpy implementation
        return [i for i in range(len(text_id))]


if __name__ == '__main__':
    # train_file = "./data/nnlm/text/librispeech.txt"
    dev_file = "./data/nnlm/text/dev.txt"
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
