#!/usr/bin/env bash

# Copyright 2021 Xiaomi Corporation (Author: Guo Liyong)
# Apache 2.0

# Example of transformer LM n-best rescoring with espnet pretrained models.

set -eou pipefail

stage=0

if [ $stage -le 0 ]; then
  # check test*.json are already generated
  echo "check data prepration"
  for test_set in test-clean test-other; do
    if [ ! -f exp/data/cuts_${test_set}.json.gz ]; then
      echo "Refer ./run.sh to generate manifest files, i.e. exp/data/*.gz."
      exit 1
    fi
  done
fi
if [ $stage -le 1 ]; then
  # Download espnet pretrained models
  # The original link of these models is:
  # https://zenodo.org/record/4604066#.YKtNrqgzZPY
  # which is accessible by espnet utils
  # The are ported to following link for users who don't have espnet dependencies.
  if [ ! -d snowfall_model_zoo ]; then
    echo "About to download pretrained models."
    git clone https://huggingface.co/GuoLiyong/snowfall_model_zoo
    ln -sf snowfall_model_zoo/exp/kamo-naoyuki/ exp/
  fi
  echo "Pretrained models are ready."

fi

if [ $stage -le 2 ]; then
  echo "Start to recognize."
  export CUDA_VISIBLE_DEVICES=3
  model_path=exp/kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave/
  python3 tokenizer_ctc_att_transformer_decode.py \
          --num_paths 100 \
          --asr_train_config $model_path/config.yaml \
          --asr_model_file $model_path/valid.acc.ave_10best.pth \
          --lm_train_config $model_path/lm/config.yaml \
          --lm_model_file $model_path/lm/valid.loss.ave_10best.pth

fi
