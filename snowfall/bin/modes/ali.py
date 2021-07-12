# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from pathlib import Path
from typing import Optional

import sys

import click
import k2
import torch

from .cli_base import cli
from snowfall.tools.ali import compute_edit_distance
from snowfall.tools.ali import convert_id_to_symbol
from snowfall.tools.ali import visualize as visualize_impl


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


@ali.command()
@click.option('-i',
              '--input',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Input sound filename.')
@click.option('-t',
              '--text-grid',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Text grid filename')
@click.option('-o',
              '--output-file',
              type=click.Path(dir_okay=False),
              required=True,
              help='Output filename. Must end with .pdf, .png, or .eps')
@click.option('-s',
              '--start',
              type=float,
              default=0.0,
              show_default=True,
              help='Start time in seconds')
@click.option('-e',
              '--end',
              type=float,
              default=0.0,
              show_default=True,
              help='End time in seconds. 0 means the end of the sound file')
@click.option('-w',
              '--width',
              type=float,
              default=6.0,
              show_default=True,
              help='The width of the viewport. Select a large '
              'value for a long sound file')
@click.option('-h',
              '--height',
              type=float,
              default=4.0,
              show_default=True,
              help='The height of the viewport')
@click.option('-f',
              '--font-size',
              type=int,
              default=12,
              show_default=True,
              help='Text font size.')
def visualize(input: str,
              text_grid: str,
              output_file: str,
              start: float = 0.0,
              end: float = 0.0,
              width: float = 6.0,
              height: float = 4.0,
              font_size: int = 12):
    '''Visualize a text grid file using Praat.

    Usage:

        snowfall ali visualize -i /path/foo.wav \
                               -t /path/foo.TextGrid \
                               -o /path/foo.pdf
    '''
    assert Path(output_file).suffix in ('.pdf', '.png', '.eps'), \
            f'It supports only pdf, png, and eps format at present. ' \
            'Given: {output_file}'

    visualize_impl(input=input,
                   text_grid=text_grid,
                   output_file=output_file,
                   start=start,
                   end=end,
                   width=width,
                   height=height,
                   font_size=font_size)

    print(f'Saved to {output_file}', file=sys.stderr)
