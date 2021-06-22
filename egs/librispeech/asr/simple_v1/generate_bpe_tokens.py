#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
'''
Example usage of this script:

python3 ./generate_bpe_tokens.py \
  --model-file ./data/lang_bpe/bpe_unigram_500.model > data/lang_bpe/bpe_unigram_500.tokens
'''

from pathlib import Path

import argparse
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file',
                        type=str,
                        help='Pre-trained BPE model file')
    args = parser.parse_args()
    model_file = args.model_file
    assert Path(model_file).is_file(), f'{model_file} does not exist'

    sp = spm.SentencePieceProcessor(model_file=model_file)
    vocab_size = sp.vocab_size()
    for i in range(vocab_size):
        print(sp.id_to_piece(i), i)


if __name__ == '__main__':
    main()
