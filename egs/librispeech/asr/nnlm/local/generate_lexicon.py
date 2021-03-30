#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import decoders


def get_args():
    parser = argparse.ArgumentParser(
        description='generate words.txt tokens.txt and lexicon.txt')
    parser.add_argument('--lexicon-path',
                        default='data/nnlm/lexicon',
                        type=str,
                        help="path to save lexicon files")
    parser.add_argument('--tokenizer-path',
                        type=str,
                        default='./data/lm_train/tokenizer-librispeech.json',
                        help="path to load tokenizer")
    parser.add_argument('--train-file',
                        default='data/nnlm/text/librispeech.txt',
                        type=str,
                        help="""file to be tokenized""")
    args = parser.parse_args()
    return args


def generate_tokens(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    symbols = tokenizer.get_vocab()
    tokens_file = '{}/tokens.txt'.format(args.lexicon_path)
    tokens_f = open(tokens_file, 'w')
    for idx, sym in enumerate(symbols):
        tokens_f.write('{} {}\n'.format(sym, idx))

    tokens_f.close()


def generate_lexicon(args, words):
    lexicon_file = '{}/lexicon.txt'.format(args.lexicon_path)
    lf = open(lexicon_file, 'w')
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.decoder = decoders.WordPiece()
    for word in words:
        output = tokenizer.encode(word)
        tokens = " ".join(output.tokens)
        lf.write("{}\t{}\n".format(word, tokens))
    lf.close()


def load_words(args):
    words = []
    tokens_file = '{}/words.txt'.format(args.lexicon_path)
    special_words = [
        '<eps>', '!SIL', '<SPOKEN_NOISE>', '<UNK>', '<s>', '</s>', '#0'
    ]

    with open(tokens_file) as f:
        for line in f:
            arr = line.strip().split()
            if arr[0] not in special_words:
                words.append(arr[0])

    return words


def main():
    args = get_args()
    generate_tokens(args)
    words = load_words(args)
    generate_lexicon(args, words)


if __name__ == '__main__':
    main()
