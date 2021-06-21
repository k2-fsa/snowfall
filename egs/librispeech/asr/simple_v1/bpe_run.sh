#!/usr/bin/env bash

# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo)
# Apache 2.0

# Example of Bpe training with ctc loss and lable smooth loss

stage=0
download_model=1
if [ $download_model -eq 1 ]; then
  echo "Use pretrained model from model zoo for decoding and evaluation"
  if [ -d snowfall_bpe_model ]; then
    echo "Model seems already been dowloaded"
  else
    git clone https://huggingface.co/GuoLiyong/snowfall_bpe_model
    for sub_dir in data exp-bpe-conformer-512-8-noam; do
      ln -sf snowfall_bpe_model/$sub_dir ./
    done
  fi
fi

if [ ! -f exp/data/cuts_test-clean.json.gz ]; then
  echo "Please follow stage 5 in ./run.sh to generate test manifests"
  exit 1
fi

if [ $stage -le 1 ]; then
  export CUDA_VISIBLE_DEVICES=3
  python bpe_ctc_att_conformer_decode.py \
    --max-duration=10 \
    --generate-release-model=False \
    --decode_with_released_model=True
fi
