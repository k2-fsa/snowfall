#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

# reference: https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
import argparse
import logging
import os
import shutil
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


def get_args():
    parser = argparse.ArgumentParser(
        description='train and tokenize with huggingface tokenizer')
    parser.add_argument('--train-file',
                        type=str,
                        help="""file to train tokenizer""")
    parser.add_argument('--vocab-size',
                        type=int,
                        default=10000,
                        help="""number of tokens of the tokenizer""")
    parser.add_argument('--tokenizer-path',
                        type=str,
                        help="path to save or load tokenizer")
    parser.add_argument('--test-file',
                        type=str,
                        help="""file to be tokenized""")
    args = parser.parse_args()
    return args


def train_tokenizer(train_files, save_path, vocab_size):
    if os.path.exists(save_path):
        logging.warning(
            "{} already exists. Please check that.".format(save_path))
        return
    else:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
    tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    # default vocab_size=30000
    # here set vocab_size=1000 for accelerating
    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=['[UNK]'])
    tokenizer.train(train_files, trainer)
    tokenizer.save(save_path)


def tokenize_text(test_file, tokenizer_path):
    if not os.path.exists(tokenizer_path):
        logging.warning(
            "Tokenizer {} does not exist. Please check that.".format(
                tokenizer_path))
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.decoder = decoders.WordPiece()
    tokenized_file = "{}.tokens".format(test_file)
    # tokenized_ids = "{}.ids".format(test_file)
    if os.path.exists(tokenized_file):
        logging.warning(
            "The input file seems already tokenized. Buckupping previous result"
        )
        shutil.copyfile(tokenized_file, "{}.bk".format(tokenized_file))
    logging.warning("Tokenizing {}.".format(test_file))
    fout = open(tokenized_file, 'w')
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            output = tokenizer.encode(line)
            fout.write(" ".join(output.tokens) + '\n')

    fout.close()


def main():
    args = get_args()
    if args.train_file is not None:
        train_files = [args.train_file]
        train_tokenizer(train_files, args.tokenizer_path, args.vocab_size)

    if args.test_file is not None:
        tokenize_text(args.test_file, args.tokenizer_path)


if __name__ == '__main__':
    main()
