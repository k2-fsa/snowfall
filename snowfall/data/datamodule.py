from torch.utils.data import DataLoader
from typing import List, Union

import argparse

from lhotse import CutSet


class DataModule:
    """
    Contains dataset-related code. It is intended to read/construct Lhotse cuts,
    and create Dataset/Sampler/DataLoader out of them.

    There is a separate method to create each of train/valid/test DataLoader.
    In principle, there might be multiple DataLoaders for each of train/valid/test
    (e.g. when a corpus has multiple test sets).
    The API of this class allows to return lists of CutSets/DataLoaders.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        pass

    def train_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def valid_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def test_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def train_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()

    def valid_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()

    def test_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()
