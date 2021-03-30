#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Liyong Guo)
# Apache 2.0

# References:
# https://github.com/kaldi-asr/kaldi/blob/master/scripts/rnnlm/train_rnnlm.sh
# https://github.com/kaldi-asr/kaldi/blob/pybind11/egs/librispeech/s5/local/rnnlm/tuning/run_tdnn_lstm_1a.sh#L75
# https://github.com/kaldi-asr/kaldi/blob/master/scripts/rnnlm/prepare_rnnlm_dir.sh
# https://github.com/pytorch/examples/tree/master/word_language_model
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

# Example of how to use HuggingFace tokenizer and train {RNN, Transformer} based LMs

set -e
stage=$1

lm_train=data/lm_train/
tokenizer=$lm_train/tokenizer-librispeech.json

text=data/local/lm/librispeech-lm-norm.txt.gz
text_dir=data/nnlm/text
train_text=$text_dir/librispeech.txt
if [ $stage -eq 0 ]; then
  mkdir -p $text_dir
  if [ ! -f $text ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm 
  fi
  echo -n >$text_dir/dev.txt
  # hold out one in every 2000 lines as dev data.
  gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$train_text
fi


if [ $stage -eq 2 ]; then
  echo "training tokenizer"
  python3 local/huggingface_tokenizer.py \
    --train-file=$train_text \
    --tokenizer-path=$tokenizer
fi

if [ $stage -eq 3 ]; then
  echo "generate lexicon"
  python local/generate_lexicon.py
fi

if [ $stage -eq 5 ]; then
  python main.py \
    --cuda \
    --model Transformer
fi
