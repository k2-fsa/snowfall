#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
'''
Example usage of this script:

python3 ./train_bpe_model.py
  --transcript data/lang_bpe/transcript.txt \
  --output-dir data/lang_bpe \
  --model-type unigram \
  --vocab-size 500

It will generate two files:
(1) data/lang_bpe/bpe_unigram_500.model
(2) data/lang_bpe/bpe_unigram_500.vocab

We only use the first file "bpe_unigram_500.model".
'''

from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import argparse
import sys

try:
    import sentencepiece as spm
except ImportError:
    print('Please run:\n\n\t'
          'pip install sentencepiece\n\n'
          'before running this script.')
    sys.exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--transcript',
                        type=str,
                        help='Path to the transcript.')

    parser.add_argument('--vocab-size',
                        type=int,
                        default=500,
                        help='vocab size for BPE training')

    parser.add_argument('--model-type',
                        type=str,
                        default='unigram',
                        choices=['unigram', 'bpe', 'word', 'char'],
                        help='model algorithm for BPE training')

    parser.add_argument('--output-dir',
                        type=str,
                        required=True,
                        help='Output directory')
    return parser


def main():
    args = get_parser().parse_args()
    assert Path(args.transcript).is_file(), f'{args.transcript} does not exist'
    assert args.model_type in ('unigram', 'bpe', 'word', 'char')
    assert args.vocab_size > 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    minloglevel = 0  # change it to 1 to disable INFO logs
    transcript = args.transcript
    vocab_size = args.vocab_size
    model_type = args.model_type
    model_prefix = f'{args.output_dir}/bpe_{model_type}_{vocab_size}'
    input_sentence_size = 100000000
    user_defined_symbols = ['<blk>']

    # By default, unk_id is 0, but we want to map <blk> to 0, so
    # We define unk_id to 1. You can choose any value (less than vocab_size),
    # for unk_id except 0, which is occupied by the above `<blk>` symbol
    unk_id = 1

    # `<blk>` is guaranteed to be mapped to 0

    spm.SentencePieceTrainer.train(input=transcript,
                                   model_prefix=model_prefix,
                                   vocab_size=vocab_size,
                                   model_type=model_type,
                                   user_defined_symbols=user_defined_symbols,
                                   unk_id=unk_id,
                                   bos_id=-1,
                                   eos_id=-1,
                                   input_sentence_size=input_sentence_size,
                                   minloglevel=minloglevel)

    print(f'Generated "{model_prefix}.model" and {model_prefix}.vocab')


if __name__ == '__main__':
    main()
