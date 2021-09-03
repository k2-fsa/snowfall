#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang
#                                            Mingshuang Luo)
#           2021 Pingfeng Luo
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

dataset_path=(
  /ceph-meixu/luomingshuang/audio-data/timit
  )

data=${dataset_path[0]}
for d in ${dataset_path[@]}; do
  if [ -d $d ]; then
    data=$d
    break
  fi
done

if [ ! -d $data ]; then
  echo "$data does not exist"
  exit 1
fi

[ -f path.sh ] && . ./path.sh

stage=1

if [ $stage -le 1 ]; then
  echo "Data preparation"
  local2/timit_data_prep.sh $data
fi


if [ $stage -le 2 ]; then
  echo "Dict preparation"
  local2/timit_prepare_dict.sh
fi


if [ $stage -le 3 ]; then
  echo "Lang preparation"
  local/prepare_lang.sh \
  --sil-prob 0.0 \
  --position-dependent-phones false \
  --num-sil-states 3 \
  data/local/dict \
  "sil" \
  data/local/lang_tmp_nosp \
  data/lang_nosp

  echo "To load L:"
  echo "Use::"
  echo "  with open('data/lang_nosp/L.fst.txt') as f:"
  echo "    Lfst = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi


if [ $stage -le 4 ]; then
  echo "LM preparation"
  local2/train_lms.sh
  gunzip -c data/local/lm/3gram-mincount/lm_unpruned.gz >data/local/lm/lm_tgmed.arpa
  # Note: you need to install kaldilm using `pip install kaldilm`
  # Build G
  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=1 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G_uni.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/local/lm/lm_tgmed.arpa >data/lang_nosp/G.fst.txt

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi


if [ $stage -le 5 ]; then
  echo "Feature preparation"
  python3 ./prepare.py
fi

if [ $stage -le 6 ]; then
  echo "Training"
  python3 ./ctc_train.py
fi

if [ $stage -le 7 ]; then
  echo "Decoding"
  python3 ./ctc_decode.py
fi

