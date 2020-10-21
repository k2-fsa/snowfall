#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -e

stage=1

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.magicdatatech.com/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  local/prepare_lang.sh data/local/dict_nosp "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

   # Use the following Python line to load L:
   # fsa = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)
fi

if [ $stage -le 4 ]; then
  # Build G
  echo "Not ready yet."
fi