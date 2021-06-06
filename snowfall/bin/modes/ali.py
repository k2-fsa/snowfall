# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from typing import Optional

import click
import k2
import torch

from .cli_base import cli
from snowfall.common import write_error_stats
from snowfall.tools.ali import convert_id_to_symbol


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
@click.option('-s',
              '--symbol-table',
              type=click.Path(exists=True, dir_okay=False),
              help='The symbol table for the type of alignment')
def edit_distance(ref: str,
                  hyp: str,
                  type: str,
                  symbol_table: Optional[str] = None):
    '''Compute edit distance between two alignments.

    The reference/hypothesis alignment file contains a python
    object Dict[str, List[Alignment]] and this file can
    be loaded using `torch.load`. The dict is indexed by utterance ID.

    The symbol table, if provided, has the following format for each line:

        symbol integer_id

    It can be loaded by `k2.SymbolTable.from_file()`
    '''
    ref_ali = torch.load(ref)
    hyp_ali = torch.load(hyp)
    print(ref_ali)
    print(hyp_ali)
    if symbol_table:
        symbols = k2.SymbolTable.from_file(symbol_table)
        ref_ali = convert_id_to_symbol(ref_ali, type, symbols)
        hyp_ali = convert_id_to_symbol(hyp_ali, type, symbols)
        print(ref_ali)
        print(hyp_ali)

    filename = 'a.txt'
    with open(filename, 'w') as f:
        # TODO(fangjun): Use write_error_stats()
        # to print the result
        pass
