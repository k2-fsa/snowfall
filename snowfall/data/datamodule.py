from torch.utils.data import DataLoader
from typing import Iterable, Union

import argparse


class DataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def train_dataloaders(self) -> Union[DataLoader, Iterable[DataLoader]]:
        raise NotImplementedError()

    def valid_dataloaders(self) -> Union[DataLoader, Iterable[DataLoader]]:
        raise NotImplementedError()

    def test_dataloaders(self) -> Union[DataLoader, Iterable[DataLoader]]:
        raise NotImplementedError()
