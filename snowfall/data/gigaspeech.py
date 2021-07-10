import argparse
import logging
from functools import lru_cache

from lhotse import CutSet, load_manifest
from snowfall.common import str2bool
from snowfall.data.asr_datamodule import AsrDataModule


def get_context_suffix(args):
    if args.context_window is None or args.context_window <= 0.0:
        ctx_suffix = ""
    else:
        ctx_suffix = f"_{args.context_direction}{args.context_window}"
    return ctx_suffix


class GigaSpeechAsrDataModule(AsrDataModule):
    """
    GigaSpeech ASR data module. Can be used for any subset.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(title="GigaSpeech specific options")
        group.add_argument(
            "--subset",
            type=str,
            default="XS",
            help="Select the GigaSpeech subset (XS|S|M|L|XL)",
        )
        group.add_argument(
            "--context-window",
            type=float,
            default=0.0,
            help="Training cut duration in seconds. "
                 "Use 0 to train on supervision segments without acoustic context, with variable cut lengths; "
                 "number larger than zero will create multi-supervisions cuts with actual acoustic context. ",
        )
        group.add_argument(
            "--context-direction",
            type=str,
            default="center",
            help="If context-window is 0, does nothing. "
                 "If it's larger than 0, determines in which direction (relative to the supervision) "
                 "to seek for extra acoustic context. Available values: (left|right|center|random).",
        )
        group.add_argument(
            '--use-context-for-test',
            type=str2bool,
            default=False,
            help='Should we read cuts with acoustic context or without it. '
                 '(note: for now, they may contain duplicated segments)'
        )

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        cuts_train = load_manifest(
            self.args.feature_dir
            / f"gigaspeech_cuts_{self.args.subset}{get_context_suffix(self.args)}.jsonl.gz"
        )
        return cuts_train

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        if self.args.use_context_for_test:
            path = self.args.feature_dir / f"gigaspeech_cuts_DEV{get_context_suffix(self.args)}.jsonl.gz"
        else:
            path = self.args.feature_dir / f"gigaspeech_cuts_DEV.jsonl.gz"
        logging.info(f"About to get valid cuts from {path}")
        cuts_valid = load_manifest(path)
        return cuts_valid

    @lru_cache()
    def test_cuts(self) -> CutSet:
        if self.args.use_context_for_test:
            path = self.args.feature_dir / f"gigaspeech_cuts_TEST{get_context_suffix(self.args)}.jsonl.gz"
        else:
            path = self.args.feature_dir / f"gigaspeech_cuts_TEST.jsonl.gz"
        logging.info(f"About to get test cuts from {path}")
        cuts_test = load_manifest(path)
        return cuts_test
