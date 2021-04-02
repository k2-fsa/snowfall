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

exp=exp-nnlm
tokenizer=$exp/tokenizer-librispeech.json

text=data/local/lm/librispeech-lm-norm.txt.gz
text_dir=data/nnlm/text
all_train_text=$text_dir/librispeech.txt
# there are 40,398,052 pieces in all_train_text, which will take 50 MINUTES to be tokenized, with a single process.
# Now only $train_pieces data is used for debugging pipeline
train_pieces=100000 # 5 times of dev.txt
# uncomment follwoing line to use all_train_text
# train_pieces=
dev_text=$text_dir/dev.txt

# vocab_size of huggingface tokenizer
vocab_size=2000
# for neural models, number of final classes is:
# ntokens = $vocab_size + 3
# while: bos_id = ntokens - 3
#        eos_id = ntokens - 2
#        pad_index = ntokens - 1


mkdir -p $text_dir

if [ $stage -eq -1 ]; then
  # env for experiment ../simple_v1 is expected to have been built.
  echo "Install extra dependencies"
  pip install -r requirements.txt
fi

if [ $stage -eq 0 ]; then
  # reference:
  # https://github.com/kaldi-asr/kaldi/blob/pybind11/egs/librispeech/s5/local/rnnlm/tuning/run_tdnn_lstm_1a.sh#L75
  # use the same data seperation method to kaldi whose result can be used as a baseline
  if [ ! -f $text ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm
  fi
  echo -n >$text_dir/dev.txt
  # hold out one in every 2000 lines as dev data.
  gunzip -c $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%2000 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$all_train_text
fi

if [ ! -z "$train_pieces" ]; then
  train_text=$text_dir/${train_pieces}_librispeech.txt
  if [ $train_text -ot $all_train_text ] || [  ! -f $train_text ]; then
  # if [ ! -f $train_text) || $train_text -ot $all_train_text ]; then
    head -n $train_pieces $all_train_text > $train_text
  fi
else
  train_text=$all_train_text
fi

if [ $stage -eq 1 ]; then
  echo "training tokenizer"
  python3 local/huggingface_tokenizer.py \
    --train-file $train_text \
    --vocab-size $vocab_size \
    --tokenizer-path $tokenizer
fi


if [ $stage -eq 2 ]; then
  echo "tokenize train and dev files"
  for text in $dev_text $train_text; do
    python3 local/huggingface_tokenizer.py \
      --test-file $text \
      --tokenizer-path $tokenizer
  done
fi

if [ $stage -eq 3 ]; then
  echo "start to train"
  # model_iter if for resume training
  # -1 means train from scratch
  python main.py \
    --model_iter 48 \
    --train_token ${train_text}.tokens \
    --vocab_size $vocab_size \
    --model_type Transformer

fi

if [ $stage -eq 4 ]; then
  # generate words.txt tokens.txt and lexicion.txt
  # which is used in future rescore process
  lexicon_path=./data/nnlm/lexicon
  mkdir -p $lexicon_path
  words_txt=../simple_v1/data/lang_nosp/words.txt
  if [ -f $words_txt ]; then
    cp $words_txt $lexicon_path
  else
    echo "please set words_txt path of your previous experiment"
    echo "the NN-LM trained LM is used as a rescore module, \
      currently the same words.txt with previous experiment is prefered"
  fi
  echo "generate lexicon"
  python local/generate_lexicon.py \
    --tokenizer-path $tokenizer

fi
