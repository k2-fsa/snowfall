# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Union

import k2

from snowfall.common import write_error_stats


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

    # We assume that the same utterance exists both in ref_ali and in hyp_ali.
    # An exception is thrown if the assumption is violated.
    for utt in ref_ali:
        ref = ref_ali[utt].value[type]
        hyp = hyp_ali[utt].value[type]
        pairs.append((ref, hyp))

    with open(output_file, 'w') as f:
        write_error_stats(f, 'test', pairs)
