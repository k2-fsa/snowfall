#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import k2
import logging
import os
import re
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from snowfall.models import AcousticModel

Pathlike = Union[str, Path]


def setup_logger(log_filename: Pathlike, log_level: str = 'info', use_console: bool = True) -> None:
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
    if use_console:
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
        'epoch': epoch,
        'learning_rate': learning_rate,
        'objf': objf,
        'valid_objf': valid_objf,
        'global_batch_idx_train': global_batch_idx_train,
    }
    if hasattr(model, 'module'):
        # Saving checkpoint from DDP model
        checkpoint.update({
            'num_features': model.module.num_features,
            'num_classes': model.module.num_classes,
            'subsampling_factor': model.module.subsampling_factor,
        })
    else:
        checkpoint.update({
            'num_features': model.num_features,
            'num_classes': model.num_classes,
            'subsampling_factor': model.subsampling_factor,
        })
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


def cut_id_dumper(dataloader, path: Path):
    """
    Debugging utility. Writes processed cut IDs to a file.
    Expects ``return_cuts=True`` to be passed to the Dataset class.

    Example::

        >>> for batch in cut_id_dumper(dataloader):
        ...     pass
    """
    if not dataloader.dataset.return_cuts:
        return dataloader  # do nothing, "return_cuts=True" was not set
    with path.open('w') as f:
        for batch in dataloader:
            for cut in batch['supervisions']['cut']:
                print(cut.id, file=f)
            yield batch
