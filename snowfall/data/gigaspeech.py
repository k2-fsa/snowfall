import argparse

from functools import lru_cache

import logging
from typing import List

from lhotse import CutSet, load_manifest
from snowfall.common import str2bool
from snowfall.data.asr_datamodule import AsrDataModule


class GigaSpeechAsrDataModule(AsrDataModule):
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(title='GigaSpeech specific options')
        group.add_argument(
            '--subset',
            type=str,
            default='{XS}',
            help='Subset of the corpus.')

    @lru_cache()
    def train_cuts(self) -> CutSet:
        return load_manifest(self.args.feature_dir / f'cuts_{self.args.subset}.jsonl.gz')

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        return load_manifest(self.args.feature_dir / 'cuts_{DEV}.jsonl.gz')

    @lru_cache()
    def test_cuts(self) -> List[CutSet]:
        return load_manifest(self.args.feature_dir / 'cuts_{TEST}.jsonl.gz')
