#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

import argparse
import logging
from typing import Tuple

import numpy as np
import torch

from espnet_utils.common import load_espnet_model_config
from espnet_utils.common import rename_state_dict, combine_qkv
from espnet_utils.frontened import Fbank
from espnet_utils.frontened import GlobalMVN
from espnet_utils.numericalizer import SpmNumericalizer
from snowfall.models.conformer import Conformer

_ESPNET_ENCODER_KEY_TO_SNOWFALL_KEY = [
    ('frontend.logmel.melmat', 'frontend.melmat'),
    ('encoder.embed.out.0.weight', 'encoder.embed.out.weight'),
    ('encoder.embed.out.0.bias', 'encoder.embed.out.bias'),
    (r'(encoder.encoders.)(\d+)(.self_attn.)linear_out([\s\S*])',
     r'\1\2\3out_proj\4'),
    (r'(encoder.encoders.)(\d+)', r'\1layers.\2'),
    (r'(encoder.encoders.layers.)(\d+)(.feed_forward.)(w_1)',
     r'\1\2.feed_forward.0'),
    (r'(encoder.encoders.layers.)(\d+)(.feed_forward.)(w_2)',
     r'\1\2.feed_forward.3'),
    (r'(encoder.encoders.layers.)(\d+)(.feed_forward_macaron.)(w_1)',
     r'\1\2.feed_forward_macaron.0'),
    (r'(encoder.encoders.layers.)(\d+)(.feed_forward_macaron.)(w_2)',
     r'\1\2.feed_forward_macaron.3'),
    (r'(encoder.embed.)([\s\S*])', r'encoder.encoder_embed.\2'),
    (r'(encoder.encoders.)([\s\S*])', r'encoder.encoder.\2'),
    (r'(ctc.ctc_lo.)([\s\S*])', r'encoder.encoder_output_layer.1.\2'),
]


class ESPnetASRModel(torch.nn.Module):

    def __init__(
        self,
        frontend: None,
        normalize: None,
        encoder: None,
    ):

        super().__init__()
        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder

    def encode(
            self, speech: torch.Tensor,
            speech_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        feats, feats_lengths = self.frontend(speech, speech_lengths)

        feats, feats_lengths = self.normalize(feats, feats_lengths)

        feats = feats.permute(0, 2, 1)

        nnet_output, _, _ = self.encoder(feats)
        nnet_output = nnet_output.permute(2, 0, 1)
        return nnet_output

    @classmethod
    def build_model(cls, asr_train_config, asr_model_file, device):
        args = load_espnet_model_config(asr_train_config)
        # {'fs': '16k', 'hop_length': 256, 'n_fft': 512}
        frontend = Fbank(**args.frontend_conf)
        normalize = GlobalMVN(**args.normalize_conf)
        encoder = Conformer(num_features=80,
                            num_classes=len(args.token_list),
                            subsampling_factor=4,
                            d_model=512,
                            nhead=8,
                            dim_feedforward=2048,
                            num_encoder_layers=12,
                            cnn_module_kernel=31,
                            num_decoder_layers=0,
                            is_espnet_structure=True)

        model = ESPnetASRModel(
            frontend=frontend,
            normalize=normalize,
            encoder=encoder,
        )

        state_dict = torch.load(asr_model_file, map_location=device)

        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith('decoder')
        }

        combine_qkv(state_dict, num_encoder_layers=11)
        rename_state_dict(rename_patterns=_ESPNET_ENCODER_KEY_TO_SNOWFALL_KEY,
                          state_dict=state_dict)

        model.load_state_dict(state_dict, strict=False)
        model = model.to(torch.device(device))

        numericalizer = SpmNumericalizer(tokenizer_type='spm',
                                         tokenizer_file=args.bpemodel,
                                         token_list=args.token_list,
                                         unk_symbol='<unk>')
        return model, numericalizer
