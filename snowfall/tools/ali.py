# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Union

import k2


@dataclass
class Alignment:
    # type of the alignment, e.g., ilabel, phone_ilabel
    type: str

    # If it contains frame-wise alignment
    # and the sampling rate is available, we can convert it
    # to CTM, like the one used in Lhotse
    value: Union[List[int], List[str]]


# The alignment in a dataset can be represented by
#
#   Dict[str, List[Alignment]]
#
# The key of the dict can the utterance ID.
# We assume that the number of types of alignments is small,
# so we use a list to represent all the possible alignments
# of an utterance. We will do a linear search with the list to find
# the given type of alignment.


def _ids_to_symbols(ids: List[int], symbol_table: k2.SymbolTable) -> List[str]:
    '''Convert a list of IDs to a list of symbols.
    '''
    return [symbol_table.get(i) for i in ids]


def convert_id_to_symbol(ali: Dict[str, List[Alignment]], type: str,
                         symbol_table: k2.SymbolTable
                        ) -> Dict[str, List[Alignment]]:
    '''Convert IDs in alignments to symbols.

    Args:
      ali:
        A dict containing the alignments indexed by utterance ID.
      type:
        The type of alignment to be converted
      symbol_table:
        A symbol table.
    '''
    ans = {}
    for key, value in ali.items():
        # value is a List[Alignment]
        ans[key] = []
        for alignment in value:
            if alignment.type != type:
                # we use a shallow copy here
                ans[key].append(alignment)
            else:
                ans[key].append(
                    Alignment(type,
                              _ids_to_symbols(alignment.value, symbol_table)))
    return ans
