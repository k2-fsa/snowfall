import argparse
import copy
import os
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from espnet_utils.common import load_espnet_model_config
from espnet_utils.text_dataset import DatasetOption, TextFileDataIterator, AuxlabelDataIterator, AbsLMDataIterator
from espnet_utils.load_lm_model import build_lm_model_from_file
from espnet_utils.numericalizer import get_numericalizer
from snowfall.models.lm_transformer import TransformerLM

# TODO(Liyong Guo): types may need to be supported ['text', 'token', 'token_id']
_TYPES_SUPPORTED = ['text_file', 'auxlabel']


def _validate_input_type(input_type: Optional[str] = None):
    # A valid input_type must be assigned from the client
    assert input_type is not None
    assert input_type in _TYPES_SUPPORTED


@dataclass(frozen=True)
class PPLResult:
    nlls: List[float]
    ntokens: int
    nwords: int

    @property
    def total_nll(self):
        return sum(self.nlls)

    @property
    def token_ppl(self):
        return np.exp(self.total_nll / self.ntokens)

    @property
    def word_ppl(self):
        return np.exp(self.total_nll / self.nwords)


class NNLMEvaluator(object):

    @torch.no_grad()
    def nll(self, text_source):
        nlls = []
        total_nll = 0.0
        total_ntokens = 0
        total_nwords = 0
        for xs_pad, target_pad, word_lens, token_lens in self.dataset(
                text_source):
            xs_pad = xs_pad.to(self.device)
            target_pad = target_pad.to(self.device)
            nll = self.lm.nll(xs_pad, target_pad, token_lens)
            nll = nll.detach().cpu().numpy().sum(1)
            nlls.extend(nll)
            total_ntokens += sum(token_lens)
            total_nwords += sum(word_lens)
        ppl_result = PPLResult(nlls=nlls,
                               ntokens=total_ntokens,
                               nwords=total_nwords)
        return ppl_result


@dataclass
class EspnetNNLMEvaluator(NNLMEvaluator):
    lm: TransformerLM
    dataset: AbsLMDataIterator
    device: Union[str, torch.device]

    @classmethod
    def build_model(cls,
                    lm_train_config,
                    lm_model_file,
                    device='cpu',
                    input_type='text_file',
                    batch_size=32,
                    numericalizer=None):
        _validate_input_type(input_type)
        lm_model_file = lm_model_file
        train_args = load_espnet_model_config(lm_train_config)

        lm_config = copy.deepcopy(train_args.lm_conf)
        lm_config['vocab_size'] = len(train_args.token_list)

        model = build_lm_model_from_file(config=lm_config,
                                         model_file=lm_model_file,
                                         model_type='espnet')
        model.to(device)

        if numericalizer is None:
            numericalizer = get_numericalizer(
                tokenizer_type='spm',
                tokenizer_file=train_args.bpemodel,
                token_list=train_args.token_list)
        dataset_option = DatasetOption(input_type=input_type,
                                       preprocessor=numericalizer)

        if input_type == 'text_file':
            dataset = TextFileDataIterator(dataset_option)
        elif input_type == 'auxlabel':
            dataset = AuxlabelDataIterator(dataset_option,
                                           numericalizer=numericalizer)

        evaluator = EspnetNNLMEvaluator(lm=model,
                                        dataset=dataset,
                                        device=device)
        return evaluator
