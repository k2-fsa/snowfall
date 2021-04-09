#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

# modified from https://github.com/k2-fsa/snowfall/blob/master/snowfall/common.py to save/load non-Acoustic Model
import logging
import os
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

Pathlike = Union[str, Path]
Info = Optional[dict]


def load_checkpoint(filename: Pathlike,
                    model: torch.nn.Module,
                    info: Info = None) -> Dict[str, Any]:
    logging.info('load checkpoint from {}'.format(filename))

    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    return checkpoint


def save_checkpoint(filename: Pathlike,
                    model: torch.nn.Module,
                    info: Info = None) -> None:
    if not os.path.exists(os.path.dirname(filename)):
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    logging.info(f'Save checkpoint to {filename}')
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if info is not None:
        checkpoint.update(info)

    torch.save(checkpoint, filename)
