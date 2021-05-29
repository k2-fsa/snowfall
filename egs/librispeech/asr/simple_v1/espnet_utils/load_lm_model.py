#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import re
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
import yaml

from snowfall.models.lm_transformer import TransformerLM
from espnet_utils.common import rename_state_dict

_ESPNET_TRANSFORMER_LM_KEY_TO_SNOWFALL_KEY = [
    (r'([\s\S]*).feed_forward.w_1', r'\1.linear1'),
    (r'([\s\S]*).feed_forward.w_2', r'\1.linear2'),
    (r'([\s\S]*).encoder.embed([\s\S]*)', r'\1.input_embed\2'),
    (r'(lm.encoder.encoders.)(\d+)', r'\1layers.\2'),
    (r'(lm.)([\s\S]*)', r'\2'),
]


def load_espnet_model(
    config: Dict,
    model_file: Union[Path, str],
):
    """This method is used to load LM model downloaded from espnet model zoo.

    Args:
        config_file: The yaml file saved when training.
        model_file: The model file saved when training.

    """
    model = TransformerLM(**config)

    assert model_file is not None, f"model file doesn't exist"
    state_dict = torch.load(model_file)

    rename_state_dict(
        rename_patterns=_ESPNET_TRANSFORMER_LM_KEY_TO_SNOWFALL_KEY,
        state_dict=state_dict)
    model.load_state_dict(state_dict)

    return model


def build_lm_model_from_file(config=None,
                             model_file=None,
                             model_type='espnet'):
    if model_type == 'espnet':
        return load_espnet_model(config, model_file)
    elif model_type == 'snowfall':
        raise NotImplementedError(f'Snowfall model to be suppported')
    else:
        raise ValueError(f'Unsupported model type {model_type}')
