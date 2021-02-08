#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional
import re

import k2
import torch

from snowfall.models import AcousticModel

Pathlike = Union[str, Path]


def setup_logger(log_filename: Pathlike, log_level: str = 'info') -> None:
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename = '{}-{}'.format(log_filename, date_time)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    formatter = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
    level = logging.ERROR
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    logging.basicConfig(filename=log_filename,
                        format=formatter,
                        level=level,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(formatter))
    logging.getLogger('').addHandler(console)


def load_checkpoint(filename: Pathlike, model: AcousticModel) -> Dict[str, Any]:
    logging.info('load checkpoint from {}'.format(filename))

    checkpoint = torch.load(filename, map_location='cpu')

    keys = [
        'state_dict', 'epoch', 'learning_rate', 'objf', 'valid_objf',
        'num_features', 'num_classes', 'subsampling_factor',
        'global_batch_idx_train'
    ]
    missing_keys = set(keys) - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f"Missing keys in checkpoint: {missing_keys}")

    if not list(model.state_dict().keys())[0].startswith('module.') \
            and list(checkpoint['state_dict'])[0].startswith('module.'):
        # the checkpoint was saved by DDP
        logging.info('load checkpoint from DDP')
        dst_state_dict = model.state_dict()
        src_state_dict = checkpoint['state_dict']
        for key in dst_state_dict.keys():
            src_key = '{}.{}'.format('module', key)
            dst_state_dict[key] = src_state_dict.pop(src_key)
        assert len(src_state_dict) == 0
        model.load_state_dict(dst_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.num_features = checkpoint['num_features']
    model.num_classes = checkpoint['num_classes']
    model.subsampling_factor = checkpoint['subsampling_factor']

    return checkpoint


def save_checkpoint(
        filename: Pathlike,
        model: AcousticModel,
        epoch: int,
        learning_rate: float,
        objf: float,
        valid_objf: float,
        global_batch_idx_train: int,
        local_rank: int = 0
) -> None:
    if local_rank is not None and local_rank != 0:
        return
    logging.info(f'Save checkpoint to {filename}: epoch={epoch}, '
                 f'learning_rate={learning_rate}, objf={objf}, valid_objf={valid_objf}')
    checkpoint = {
        'state_dict': model.state_dict(),
        'num_features': model.num_features,
        'num_classes': model.num_classes,
        'subsampling_factor': model.subsampling_factor,
        'epoch': epoch,
        'learning_rate': learning_rate,
        'objf': objf,
        'valid_objf': valid_objf,
        'global_batch_idx_train': global_batch_idx_train,
    }
    torch.save(checkpoint, filename)


def save_training_info(
        filename: Pathlike,
        model_path: Pathlike,
        current_epoch: int,
        learning_rate: float,
        objf: float,
        best_objf: float,
        valid_objf: float,
        best_valid_objf: float,
        best_epoch: int,
        local_rank: int = 0
):
    if local_rank is not None and local_rank != 0:
        return

    with open(filename, 'w') as f:
        f.write('model_path: {}\n'.format(model_path))
        f.write('epoch: {}\n'.format(current_epoch))
        f.write('learning rate: {}\n'.format(learning_rate))
        f.write('objf: {}\n'.format(objf))
        f.write('best objf: {}\n'.format(best_objf))
        f.write('valid objf: {}\n'.format(valid_objf))
        f.write('best valid objf: {}\n'.format(best_valid_objf))
        f.write('best epoch: {}\n'.format(best_epoch))

    logging.info('write training info to {}'.format(filename))


def get_phone_symbols(symbol_table: k2.SymbolTable,
                      pattern: str = r'^#\d+$') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.

    Caution:
      0 is not a phone ID so it is excluded from the return value.

    Args:
      symbol_table:
        A symbol table in k2.
      pattern:
        Symbols containing this pattern are disambiguation symbols.
    Returns:
      Return a list of symbol IDs excluding those from disambiguation symbols.
    '''
    regex = re.compile(pattern)
    symbols = symbol_table.symbols
    ans = []
    for s in symbols:
        if not regex.match(s):
            ans.append(symbol_table[s])
    if 0 in ans:
        ans.remove(0)
    ans.sort()
    return ans


def describe(model: torch.nn.Module):
    print('=' * 80)
    print('Model parameters summary:')
    print('=' * 80)
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        print(f'* {name}: {num_params:>{80 - len(name) - 4}}')
    print('=' * 80)
    print('Total:', total)
    print('=' * 80)


def get_texts(best_paths: k2.Fsa, indices: Optional[torch.Tensor] = None) -> List[List[int]]:
    '''Extract the texts from the best-path FSAs, in the original order (before
       the permutation given by `indices`).
       Args:
           best_paths:  a k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
                    containing multiple FSAs, which is expected to be the result
                    of k2.shortest_path (otherwise the returned values won't
                    be meaningful).  Must have the 'aux_labels' attribute, as
                  a ragged tensor.
           indices: possibly a torch.Tensor giving the permutation that we used
                    on the supervisions of this minibatch to put them in decreasing
                    order of num-frames.  We'll apply the inverse permutation.
                    Doesn't have to be on the same device as `best_paths`
      Return:
          Returns a list of lists of int, containing the label sequences we
          decoded.
    '''
    # remove any 0's or -1's (there should be no 0's left but may be -1's.)
    aux_labels = k2.ragged.remove_values_leq(best_paths.aux_labels, 0)
    aux_shape = k2.ragged.compose_ragged_shapes(best_paths.arcs.shape(),
                                                aux_labels.shape())
    # remove the states and arcs axes.
    aux_shape = k2.ragged.remove_axis(aux_shape, 1)
    aux_shape = k2.ragged.remove_axis(aux_shape, 1)
    aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    assert(aux_labels.num_axes() == 2)
    aux_labels, _ = k2.ragged.index(aux_labels,
                                    invert_permutation(indices).to(dtype=torch.int32,
                                                                   device=best_paths.device))
    return k2.ragged.to_list(aux_labels)


def invert_permutation(indices: torch.Tensor) -> torch.Tensor:
    ans = torch.zeros(indices.shape, device=indices.device, dtype=torch.long)
    ans[indices] = torch.arange(0, indices.shape[0], device=indices.device)
    return ans


def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))