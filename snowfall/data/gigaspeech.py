import argparse
import logging
from functools import lru_cache

from lhotse import CutSet, load_manifest
from snowfall.data.asr_datamodule import AsrDataModule


class GigaSpeechAsrDataModule(AsrDataModule):
    """
    GigaSpeech ASR data module. Can be used for any subset.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(title='GigaSpeech specific options')
        group.add_argument(
            "--subset",
            type=str,
            default="XS",
            help="Select the GigaSpeech subset (XS|S|M|L|XL)",
        )

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        cuts_train = load_manifest(self.args.feature_dir / f'cuts_gigaspeech_{self.args.subset}.jsonl.gz')
        return cuts_train

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        cuts_valid = load_manifest(self.args.feature_dir / 'cuts_gigaspeech_DEV.jsonl.gz')
        return cuts_valid

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        cuts_test = load_manifest(self.args.feature_dir / 'cuts_gigaspeech_TEST.jsonl.gz')
        return cuts_test
