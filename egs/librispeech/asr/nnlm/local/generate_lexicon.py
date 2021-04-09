#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import argparse
import collections
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
    ''' Extract symbols and the corresponding ids from a tokenizer,
        and save as tokens.txt.
        A real token.txt with nvocab=10000 is:
        [unk] 0
        ' 1
        a 2
        b 3
        c 4
        ...
        patty 9994
        neatly 9995
        stormy 9996
        daddy 9997
        ##enon 9998
        remarkably 9999
    '''

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    symbols = tokenizer.get_vocab()
    tokens_file = '{}/tokens.txt'.format(args.lexicon_path)
    tokens_f = open(tokens_file, 'w')
    id2sym = {idx: sym.lower() for sym, idx in symbols.items()}
    for idx in range(len(symbols)):
        assert idx in id2sym
        tokens_f.write('{} {}\n'.format(id2sym[idx], idx))

    tokens_f.close()


def generate_lexicon(args, words):
    ''' Tokenize every word in words.txt and save as lexicont.txt. 
        Each line represents a word and its tokenized representation, i.e. a sequence of tokens. a word and its tokens are seprated by a table.
 
        An example file looks like:

        abbreviating	abb ##re ##via ##ting
        abbreviation	abb ##re ##via ##t ##ion
        abbreviations	abb ##re ##via ##t ##ions
 
    '''
    special_words = [
        '<eps>', '!SIL', '<SPOKEN_NOISE>', '<UNK>', '<s>', '</s>', '#0'
    ]
    lexicon_file = '{}/lexicon.txt'.format(args.lexicon_path)
    lf = open(lexicon_file, 'w')
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.decoder = decoders.WordPiece()
    for word in words:
        if not (word.upper() in special_words or
                word.lower() in special_words):
            output = tokenizer.encode(word)
            tokens = ' '.join(output.tokens)
        else:
            tokens = '[unk]'
        lf.write("{}\t{}\n".format(word.lower(), tokens.lower()))
    lf.close()


def load_words(args):
    words = []
    tokens_file = '{}/words.txt'.format(args.lexicon_path)

    with open(tokens_file) as f:
        for line in f:
            arr = line.strip().split()
            words.append(arr[0].lower())

    return words


def main():
    args = get_args()
    generate_tokens(args)
    words = load_words(args)
    generate_lexicon(args, words)


if __name__ == '__main__':
    main()
