# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from pathlib import Path

import click
import torch

from .cli_base import cli
from snowfall.tools.net import compute_post as compute_post_impl


@cli.group()
def net():
    '''
    Neural network tools in snowfall.
    '''
    pass


@net.command()
@click.option('-m',
              '--model',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Path to Torch Scripted module')
@click.option('-f',
              '--feats',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Path to Featureset manifest')
@click.option('-o',
              '--output-dir',
              type=click.Path(dir_okay=True),
              required=True,
              help='Output directory')
@click.option('-l',
              '--max-duration',
              default=200,
              type=int,
              show_default=True,
              help='max duration in seconds in a batch')
@click.option('-i',
              '--device-id',
              default=0,
              type=int,
              show_default=True,
              help='-1 to use CPU. Otherwise, it is the GPU device ID')
def compute_post(model: str,
                 feats: str,
                 output_dir: str,
                 max_duration: int = 200,
                 device_id: int = 0):
    '''Compute posteriors given a model and a FeatureSet.
    '''
    if device_id < 0:
        print('Use CPU')
        device = torch.device('cpu')
    else:
        print(f'Use GPU {device_id}')
        device = torch.device('cuda', device_id)

    compute_post_impl(model=model,
                      feats=feats,
                      output_dir=output_dir,
                      max_duration=max_duration,
                      device=device)
