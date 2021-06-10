# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Union
from pathlib import Path

import os
import shutil
import sys
import tempfile

import k2
import lhotse
import torch

from snowfall.common import find_first_disambig_symbol
from snowfall.common import invert_permutation
from snowfall.common import write_error_stats
from snowfall.decoding.graph import compile_HLG
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import get_phone_symbols


@dataclass
class Alignment:
    # The key of the dict indicates the type of the alignment,
    # e.g., ilabel, phone_label, etc.
    #
    # The value of the dict is the actual alignments.
    # If the alignments are frame-wise and if the sampling rate
    # is available, they can be converted to CTM format, like the
    # one used in Lhotse
    value: Dict[str, Union[List[int], List[str]]]


# The alignment in a dataset can be represented by
#
#   Dict[str, Alignment]
#
# The key of the dict can be utterance IDs.


def _ids_to_symbols(ids: List[int], symbol_table: k2.SymbolTable) -> List[str]:
    '''Convert a list of IDs to a list of symbols.
    '''
    return [symbol_table.get(i) for i in ids]


def convert_id_to_symbol(ali: Dict[str, Alignment], type: str,
                         symbol_table: k2.SymbolTable) -> None:
    '''Convert IDs in alignments to symbols.

    Caution:
      `ali` is changed in-place.

    Args:
      ali:
        A dict containing the alignments indexed by utterance ID.
      type:
        The type of alignment to be converted
      symbol_table:
        A symbol table.
    Returns:
      Return None.
    '''
    for _, utt_ali in ali.items():
        for t in utt_ali.value:
            if type == t:
                utt_ali.value[t] = _ids_to_symbols(utt_ali.value[t],
                                                   symbol_table)


def compute_edit_distance(ref_ali: Dict[str, Alignment],
                          hyp_ali: Dict[str, Alignment], type: str,
                          output_file: str) -> None:
    '''
    Args:
      ref_ali:
        The reference alignment.
      hyp_ali:
        The hypothesis alignment.
      type:
        The type of alignment to use.
      output_file:
        The filename of the output file.
    Returns:
      Return None.
    '''
    pairs = []  # each element contains a pair (ref_transcript, hyp_transcript)

    utts_in_ref = set(ref_ali.keys())
    utts_in_hyp = set(hyp_ali.keys())

    utts_in_both = utts_in_ref & utts_in_hyp

    utts_in_ref_only = utts_in_ref - utts_in_hyp
    if utts_in_ref_only:
        s = 'Decoding results are missing for the following utterances:\n\n'
        sep = '\n'
        s += f'{sep.join(utts_in_ref_only)}\n'
        print(s, file=sys.stderr)

    utts_in_hyp_only = utts_in_hyp - utts_in_ref
    if utts_in_hyp_only:
        s = 'Reference transcripts are missing for the following utterances:\n\n'
        sep = '\n'
        s += f'{sep.join(utts_in_hyp_only)}\n'
        print(s, file=sys.stderr)

    for utt in utts_in_both:
        ref = ref_ali[utt].value[type]
        hyp = hyp_ali[utt].value[type]
        pairs.append((ref, hyp))

    with open(output_file, 'w') as f:
        write_error_stats(f, 'test', pairs)


def visualize(input: str,
              text_grid: str,
              output_file: str,
              start: float = 0.0,
              end: float = 0.0,
              width: float = 6.0,
              height: float = 4.0,
              font_size: int = 12):
    '''Visualize a text gird file using Praat.

    Args:
      input:
        The filename of the input sound.
      text_grid:
        The filename of the text grid.
      output_file:
        Filename of the output file. Currently, it requires that the name ends
        with `.pdf`.
      start:
        The start time in seconds.
      end:
        The end time in seconds. 0 means the end of wave.
      width:
        The size of the viewport in inches. Select a large value if the wave is
        long.
      height:
        The size of the viewport in inches.
      font_size:
        Size of the font.
    Returns:
      Return None.
    '''
    # From https://www.fon.hum.uva.nl/praat/manual/Scripting_6_9__Calling_from_the_command_line.html
    paths = [
        r'C:\Program Files\Praat.exe',  # windows
        '/Applications/Praat.app/Contents/MacOS/Praat',  # macOS
        '/usr/bin/praat',  # Linux
    ]
    p = None
    for i in range(len(paths)):
        if Path(paths[i]).exists():
            p = paths[i]
            break
    if not p:
        p = shutil.which('praat')
    if not p:
        raise Exception('Could not find "praat". Please visit \n'
                        'https://www.fon.hum.uva.nl/praat/'
                        '\nto download it')

    suffix = Path(output_file).suffix
    assert suffix in ('.pdf', '.png', '.eps'), \
            f'It supports only pdf, png, and eps format at present. ' \
            'Given: {output_file}'
    if suffix == '.pdf':
        out_file_cmd = f'Save as PDF file: "{Path(output_file).resolve()}"'
    elif suffix == '.png':
        out_file_cmd = f'Save as 300-dpi PNG file: "{Path(output_file).resolve()}"'
    elif suffix == '.eps':
        out_file_cmd = f'Save as EPS file: "{Path(output_file).resolve()}"'
    else:
        raise ValueError(f'Unsupported extension {suffix}.\n' \
                'Only .pdf, .eps, and .png are supported')

    command = f'''
      Read from file: "{Path(input).resolve()}"
      Read from file: "{Path(text_grid).resolve()}"
      selectObject: 1, 2
      Font size: {font_size}
      Select outer viewport: 0, {width}, 0, {height}
      Draw: {start}, {end}, "yes", "yes", "yes"
      {out_file_cmd}
    '''
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()
    tmp_name = tmp_file.name
    with open(tmp_name, 'w') as f:
        f.write(command)

    cmd = f'{p} --run {tmp_name}'
    ret = os.system(cmd)
    Path(tmp_name).unlink()
    if ret != 0:
        raise Exception(f'Failed to run\n{cmd}\n'
                        f'The praat script content is:\n{command}')


def compute_ali(lang_dir: str,
                posts: str,
                output_dir: str,
                device: torch.device,
                max_duration: int = 200,
                output_beam_size: float = 8.0):
    # First, load lang
    lang_dir = Path(lang_dir)

    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)

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

    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    # Second, load posteriors
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

    ans = {}
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f'Processing batch {batch_idx}')

        nnet_output = batch['inputs'].to(device)

        supervisions = batch['supervisions']

        supervision_segments = torch.stack(
            (supervisions['sequence_idx'], supervisions['start_frame'],
             supervisions['num_frames']), 1).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        texts = supervisions['text']  # TODO: remove it

        sf = supervisions['cut'][0].posts.subsampling_factor
        dense_fsa_vec = k2.DenseFsaVec(nnet_output,
                                       supervision_segments,
                                       allow_truncate=sf - 1)

        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0,
                                             output_beam_size, 30, 10000)

        best_paths = k2.shortest_path(lattices, use_double_scores=True)
        inverted_indices = invert_permutation(indices).to(device=device,
                                                          dtype=torch.int32)

        best_paths = k2.index_fsa(best_paths, inverted_indices)

        labels = k2.RaggedInt(best_paths.arcs.shape(),
                              best_paths.labels.clone())
        labels = k2.ragged.remove_axis(labels, 1)
        labels = k2.ragged.to_list(labels)

        cut = supervisions['cut']
        for utt_ali, c in zip(labels, cut):
            ali = Alignment({})
            ali.value['phone'] = utt_ali
            # TODO(fangjun): Compute ali.value['word']
            ans[c.id] = ali

    output_dir = Path(output_dir)
    out_file = output_dir / 'ali.pt'
    torch.save(ans, out_file)

    print(f'Saved to {out_file}')
