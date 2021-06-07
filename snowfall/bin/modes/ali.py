# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from typing import Optional
import sys

import click
import k2
import torch

from .cli_base import cli
from snowfall.tools.ali import compute_edit_distance
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
@click.option('-o',
              '--output-file',
              type=click.Path(dir_okay=False),
              required=True,
              help='Output file')
@click.option('-s',
              '--symbol-table',
              type=click.Path(exists=True, dir_okay=False),
              help='The symbol table for the given type of alignment')
def edit_distance(ref: str,
                  hyp: str,
                  type: str,
                  output_file: str,
                  symbol_table: Optional[str] = None):
    '''Compute edit distance between two alignments.

    The reference/hypothesis alignment file contains a python
    object Dict[str, Alignment] and it can be loaded using
    `torch.load`. The dict is indexed by utterance ID.

    The symbol table, if provided, has the following format for each line:

        symbol integer_id

    It can be loaded by `k2.SymbolTable.from_file()`.
    '''
    ref_ali = torch.load(ref)
    hyp_ali = torch.load(hyp)

    if symbol_table:
        symbols = k2.SymbolTable.from_file(symbol_table)
        convert_id_to_symbol(ali=ref_ali, type=type, symbol_table=symbols)
        convert_id_to_symbol(ali=hyp_ali, type=type, symbol_table=symbols)

    compute_edit_distance(ref_ali=ref_ali,
                          hyp_ali=hyp_ali,
                          type=type,
                          output_file=output_file)

    print(f'Saved to {output_file}', file=sys.stderr)
