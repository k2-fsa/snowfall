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
    for sub_dir in data; do
      ln -sf snowfall_bpe_model/$sub_dir ./
    done
  fi
fi

if [ ! -f exp/data/cuts_test-clean.json.gz ]; then
  echo "Please follow stage 5 in ./run.sh to generate test manifests"
  exit 1
fi

if [ $stage -le 1 ]; then
  export CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7"
  python3 bpe_ctc_att_conformer_train.py \
    --start-epoch 0 \
    --bucketing-sampler True \
    --num-buckets 1000 \
    --lr-factor 5.0 \
    --num-epochs 50 \
    --full-libri True \
    --max-duration 200 \
    --concatenate-cuts False \
    --world-size 8

fi

if [ $stage -eq 2 ]; then
  export CUDA_VISIBLE_DEVICES=3
  python bpe_ctc_att_conformer_decode.py \
    --max-duration=10 \
    --generate-release-model=False \
    --decode_with_released_model=True
fi
