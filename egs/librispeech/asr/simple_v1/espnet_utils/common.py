#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import re
import yaml

from typing import List, Tuple, Dict
from pathlib import Path

import torch


def load_espnet_model_config(config_file):
    config_file = Path(config_file)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return argparse.Namespace(**args)


def rename_state_dict(rename_patterns: List[Tuple[str, str]],
                      state_dict: Dict[str, torch.Tensor]):
    # Rename state dict to load espent model
    if rename_patterns is not None:
        for old_pattern, new_pattern in rename_patterns:
            old_keys = [
                k for k in state_dict if re.match(old_pattern, k) is not None
            ]
            for k in old_keys:
                v = state_dict.pop(k)
                new_k = re.sub(old_pattern, new_pattern, k)
                state_dict[new_k] = v


def combine_qkv(state_dict: Dict[str, torch.Tensor], num_encoder_layers=11):
    for layer in range(num_encoder_layers + 1):
        q_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_q.weight']
        k_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_k.weight']
        v_w = state_dict[f'encoder.encoders.{layer}.self_attn.linear_v.weight']
        q_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_q.bias']
        k_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_k.bias']
        v_b = state_dict[f'encoder.encoders.{layer}.self_attn.linear_v.bias']

        for param_type in ['weight', 'bias']:
            for layer_type in ['q', 'k', 'v']:
                key_to_remove = f'encoder.encoders.{layer}.self_attn.linear_{layer_type}.{param_type}'
                state_dict.pop(key_to_remove)

        in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)
        in_proj_bias = torch.cat([q_b, k_b, v_b], dim=0)
        key_weight = f'encoder.encoders.{layer}.self_attn.in_proj.weight'
        state_dict[key_weight] = in_proj_weight
        key_bias = f'encoder.encoders.{layer}.self_attn.in_proj.bias'
        state_dict[key_bias] = in_proj_bias
