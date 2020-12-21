#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import re

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

from snowfall.models import AcousticModel
from k2 import Fsa, SymbolTable

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


def load_checkpoint(filename: Pathlike,
                    model: AcousticModel) -> Tuple[int, float, float]:
    logging.info('load checkpoint from {}'.format(filename))

    checkpoint = torch.load(filename, map_location='cpu')

    keys = [
        'state_dict', 'epoch', 'learning_rate', 'objf', 'num_features',
        'num_classes', 'subsampling_factor'
    ]
    missing_keys = set(keys) - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(f'Missing keys in checkpoint: {missing_keys}')

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

    epoch = checkpoint['epoch']
    learning_rate = checkpoint['learning_rate']
    objf = checkpoint['objf']

    return epoch, learning_rate, objf


def save_checkpoint(filename: Pathlike,
                    model: AcousticModel,
                    epoch: int,
                    learning_rate: float,
                    objf: float,
                    local_rank: int = 0) -> None:
    if local_rank is not None and local_rank != 0:
        return
    logging.info('Save checkpoint to {filename}: epoch={epoch}, '
                 'learning_rate={learning_rate}, objf={objf}'.format(
                     filename=filename,
                     epoch=epoch,
                     learning_rate=learning_rate,
                     objf=objf))
    checkpoint = {
        'state_dict': model.state_dict(),
        'num_features': model.num_features,
        'num_classes': model.num_classes,
        'subsampling_factor': model.subsampling_factor,
        'epoch': epoch,
        'learning_rate': learning_rate,
        'objf': objf
    }
    torch.save(checkpoint, filename)


def save_training_info(filename: Pathlike,
                       model_path: Pathlike,
                       current_epoch: int,
                       learning_rate: float,
                       objf: float,
                       best_objf: float,
                       best_epoch: int,
                       local_rank: int = 0):
    if local_rank is not None and local_rank != 0:
        return

    with open(filename, 'w') as f:
        f.write('model_path: {}\n'.format(model_path))
        f.write('epoch: {}\n'.format(current_epoch))
        f.write('learning rate: {}\n'.format(learning_rate))
        f.write('objf: {}\n'.format(objf))
        f.write('best objf: {}\n'.format(best_objf))
        f.write('best epoch: {}\n'.format(best_epoch))

    logging.info('write training info to {}'.format(filename))


def read_lexicon(filename: str) -> Dict[str, List[List[str]]]:
    '''Read a lexicon file.

    Args:
      filename:
        Filename of the lexicon file. Every line in the file has
        the following format::

            word phone1 phone2 phone3 phoneN

        For instance, an example file looks like the following::

            HELLO HH AH0 L OW1
            WORLD W ER1 L D

        Columns are separated by space(s), tab(s), or both.
        The first column is the word and the remaining columns
        are the pronunciations of the word.

        The second column can be pronunciation probabilities
        of this word, which is discarded.

    Returns:
      Return a dict whose key is a word and the value is a list of
      list-of-phones of the word.

      For example, if the lexicon file contains::

            HELLO HH AH0 L OW1
            HELLO HH EH0 L OW1

      The returned dict would be::

        {'HELLO': [['HH', 'AH0', 'L', 'OW1'], ['HH', 'AH0', 'L', 'OW1']]}

      Note:
        We support words with multiple pronunciations.
    '''
    ans: Dict[str, List[List[str]]] = dict()
    has_prob = None
    with open(filename, 'r', encoding='latin-1') as f:
        whitespace = re.compile('[ \t]+')
        for line in f:
            a = whitespace.split(line.strip(' \t\r\n'))
            if len(a) < 2:
                print(f'Found bad line: {line} in lexicon file {filename}')
                sys.exit(1)
            word = a[0]
            phones = a[1:]
            if has_prob is None:
                try:
                    p = float(phones[0])
                    has_prob = True
                except:
                    has_prob = False
            if has_prob:
                phones = phones[1:]

            if word not in ans:
                ans[word] = [phones]
            else:
                ans[word].append(phones)
    return ans


def build_ctc_graph(lexicon: Dict[str, List[List[str]]], text: str,
                    phone_symbol_table: SymbolTable,
                    word_symbol_table: SymbolTable):
    '''
    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
      text:
        The transcript. It contains words separated by spaces.
      phone_symbol_table:
        Phone symbol table.
      word_symbol_table:
        Word symbol table.
    '''
    oov = '<UNK>'
    assert oov in lexicon

    prev_state = 0
    whitespace = re.compile('[ \t]+')
    words = whitespace.split(text.strip(' \t\r\n'))
    rules = ''
    for word in words:
        if word not in lexicon:
            word = oov

        phones = lexicon[word]
        phones = phones[0]  # TODO(fangjun): handle multiple pronunciations
        for i, phone in enumerate(phones):
            phone_id = phone_symbol_table.get(phone)
            if i == 0:
                word_id = word_symbol_table.get(word)
            else:
                word_id = 0
            blank_state = prev_state + 1
            nonblank_state = blank_state + 1
            rules += f'{prev_state} {blank_state} 0 0 0.0\n'
            rules += f'{prev_state} {nonblank_state} {phone_id} {word_id} 0.0\n'
            rules += f'{blank_state} {blank_state} 0 0 0.0\n'
            rules += f'{blank_state} {nonblank_state} {phone_id} {word_id} 0.0\n'
            rules += f'{nonblank_state} {nonblank_state} {phone_id} 0 0.0\n'
            prev_state = nonblank_state
    final_state = prev_state + 1
    rules += f'{prev_state} {final_state} -1 -1 0.0\n'
    rules += f'{final_state}\n'
    return Fsa.from_str(rules)
