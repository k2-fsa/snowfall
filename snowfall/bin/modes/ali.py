# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from pathlib import Path
from typing import Optional

import sys

import click
import k2
import torch
import lhotse

from .cli_base import cli
from snowfall.tools.ali import compute_edit_distance
from snowfall.tools.ali import convert_id_to_symbol
from snowfall.tools.ali import visualize as visualize_impl
from snowfall.training.mmi_graph import get_phone_symbols
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.common import find_first_disambig_symbol
from snowfall.decoding.graph import compile_HLG
from snowfall.common import get_texts
from snowfall.common import store_transcripts
from snowfall.common import write_error_stats


@cli.group()
def ali():
    '''
    Alignment tools in snowfall.
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


@ali.command()
@click.option('-l',
              '--lang-dir',
              type=click.Path(exists=True, dir_okay=True, file_okay=False),
              required=True,
              help='The language dir. It is expected to '
              'contain the following files:\n'
              ' - words.txt\n'
              ' - phones.txt\n'
              ' - HLG.pt (or L_disambig.fst.txt, G.fst.txt\n')
@click.option('-p',
              '--posts',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Path to Posteriors manifest')
def compute_ali(lang_dir: str, posts: str):
    max_duration = 200
    output_beam_size = 8.0
    output_dir = Path('exp')

    lang_dir = Path(lang_dir)
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)

    phone_ids_with_blank = [0] + phone_ids
    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

    if not (lang_dir / 'HLG.pt').exists():
        print('Loading L_disambig.fst.txt')
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        print('Loading G.fst.txt')
        with open(lang_dir / 'G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(
            phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        HLG = compile_HLG(L=L,
                          G=G,
                          H=ctc_topo,
                          labels_disambig_id_start=first_phone_disambig_id,
                          aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')
    else:
        print('Loading pre-compiled HLG')
        d = torch.load(lang_dir / 'HLG.pt')
        HLG = k2.Fsa.from_dict(d)

    device = torch.device('cuda', 0)
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    cuts = lhotse.load_manifest(posts)

    dataset = lhotse.dataset.K2SpeechRecognitionDataset(
        cuts,
        input_strategy=lhotse.dataset.PrecomputedPosteriors(),
        return_cuts=True)

    sampler = lhotse.dataset.SingleCutSampler(cuts, max_duration=max_duration)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             sampler=sampler,
                                             num_workers=1)
    results = []  # a list of pair (ref_words, hyp_words)
    for batch in dataloader:
        nnet_output = batch['inputs'].to(device)

        supervisions = batch['supervisions']
        sf = supervisions['cut'][0].posts.subsampling_factor

        supervision_segments = torch.stack(
            (supervisions['sequence_idx'], supervisions['start_frame'],
             supervisions['num_frames']), 1).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        texts = supervisions['text']

        dense_fsa_vec = k2.DenseFsaVec(nnet_output,
                                       supervision_segments,
                                       allow_truncate=sf - 1)

        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0,
                                             output_beam_size, 30, 10000)

        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        assert best_paths.shape[0] == len(texts)
        hyps = get_texts(best_paths, indices)
        assert len(hyps) == len(texts)
        for i in range(len(texts)):
            hyp_words = [symbol_table.get(x) for x in hyps[i]]
            ref_words = texts[i].split(' ')
            results.append((ref_words, hyp_words))

    recog_path = output_dir / f'recogs.txt'
    store_transcripts(path=recog_path, texts=results)
    print(f'The transcripts are stored in {recog_path}')

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = output_dir / f'errs.txt'
    with open(errs_filename, 'w') as f:
        write_error_stats(f, 'compute-ali', results)
    print('Wrote detailed error stats to {}'.format(errs_filename))
