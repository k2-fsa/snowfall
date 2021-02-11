#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
#           2021 Pingfeng Luo
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

data=/mnt/cfs2/asr/database/AM/aishell
stage=1

[ -f path.sh ] && . ./path.sh

if [ $stage -le 1 ]; then
  local2/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript
fi

if [ $stage -le 2 ]; then
  local2/aishell_prepare_dict.sh $data/resource_aishell data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  local/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp || exit 1

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  local2/aishell_train_lms.sh
  gunzip -c data/local/lm/3gram-mincount/lm_unpruned.gz >data/local/lm/lm_tgmed.arpa
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
  python3 ./prepare.py
fi

if [ $stage -le 6 ]; then
  python3 ./ctc_train.py
  #python3 ./mmi_bigram_train.py
fi

if [ $stage -le 7 ]; then
  python3 ./ctc_decode.py
  #python3 ./mmi_bigram_decode.py
fi
