#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CollateFunc(object):
    '''Collate function for LMDataset
    '''

    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):
        # xs: input sequence
        # ys: label sequence
        xs = batch
        # ys = batch
        xs_pad = pad_sequence(
            [torch.from_numpy(x).int() for x in xs, True, self.pad_index])
        ys_pad = xs_pad
        return xs_pad, ys_pad


class LMDataset(Dataset):

    def __init__(self, text_file: str):
        '''Dataset to load Language Model train/dev text data

        Args:
            text_file: text file, one utt per line.
        '''
        assert os.path.exists(
            text_file), "text_file: {} does not exist, please check that."
        self.data = []
        with open(text_file, 'r') as f:
            for line in f:
                text = line.strip().split()
                assert len(text) > 0
                text_id = text2id(text)
                token_id = text_id2token_id(text_id)
                self.data.append(token_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def text2id(text: list[str]) -> list[int]:
        pass

    def text_id2token_id(text_id: list[int]) -> list[int]:
        pass


if __name__ == '__main__':
    train_file = "./data/local/lm/train.txt"
    # dev_file = "./data/local/lm/dev.txt"
    dataset = LMDataset(train_file)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=0,
                             collaate_fn=collate_func)
    for i, batch in enumerate(data_loader):
        print(i)
        print(batch)
