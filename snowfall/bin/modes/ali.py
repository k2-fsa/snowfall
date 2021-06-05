# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

import click

import torch

from .cli_base import cli


@cli.group()
def ali():
    '''
    Alignment tools in snowfall
    '''
    pass


@ali.command()
@click.option('-r',
              '--ref',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='The file containing reference alignments')
@click.option('-h',
              '--hyp',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='The file containing hypothesis alignments')
@click.option('-t',
              '--type',
              type=str,
              required=True,
              help='The type of the alignment to use for computing'
              ' the edit distance')
def edit_distance(ref: str, hyp: str, type: str):
    '''Compute edit distance between two alignments.

    The reference/hypothesis alignment file contains a python
    object Dict[str, Dict[str, List[int]]] and this file can
    be loaded using `torch.load`. The key of the dict indicates
    the utternace. The key of the second dict contains the type of
    the alignment, e.g., `ilabel`, `olabel`, and `phone_label`.
    '''
    ref_ali = torch.load(ref)
    hyp_ali = torch.load(hyp)
    print(ref_ali)
    print(hyp_ali)
