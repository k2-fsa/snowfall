#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Liyong Guo)
# Apache 2.0

# References:
# https://github.com/kaldi-asr/kaldi/blob/master/scripts/rnnlm/train_rnnlm.sh
# https://github.com/kaldi-asr/kaldi/blob/master/scripts/rnnlm/prepare_rnnlm_dir.sh
# https://github.com/pytorch/examples/tree/master/word_language_model
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

# Example of how to use HuggingFace tokenizer and train {RNN, Transformer} based LMs

set -e
stage=$1

lm_train=data/lm_train/
full_text=$lm_train/librispeech_train_960_text
tokenizer=$lm_train/tokenizer-librispeech_train_960.json
if [ $stage -eq 1 ]; then
  python3 ./local/download_lm_train_data.py
fi
if [ $stage -eq 2 ]; then
  echo "training tokenizer"
  python3 local/huggingface_tokenizer.py \
    --train-file=$full_text \
    --tokenizer-path=$tokenizer
fi

if [ $stage -eq 3 ]; then
  echo "tokenize a file"
  python3 local/huggingface_tokenizer.py \
    --test-file=$full_text \
    --tokenizer-path=$tokenizer
fi

if [ $stage -eq 4 ]; then
  echo "split all data into train/valid/test"

  full_tokens=${full_text}.tokens
  valid_test_fraction=10 # currently 5 percent for valid and 5 percent for test
  valid_test_tokens=$lm_train/valid_test.tokens
  train_tokens=$lm_train/train.tokens

  num_utts_total=$(wc -l <$full_tokens )
  num_valid_test=$(($num_utts_total/${valid_test_fraction}))
  set +x
  shuf -n $num_valid_test  $full_tokens > $valid_test_tokens

  comm -3 <(sort $valid_test_tokens) <(sort $full_tokens) > $train_tokens
  shuf -n $(($num_valid_test/2)) $valid_test_tokens > $lm_train/valid.tokens
  comm -3 <(sort $lm_train/valid.tokens) <(sort $valid_test_tokens) > $lm_train/test.tokens

fi


if [ $stage -eq 5 ]; then
  python main.py \
    --cuda \
    --model Transformer
fi
