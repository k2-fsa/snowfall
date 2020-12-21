#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

stage=1

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.magicdatatech.com/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_nosp \
    "<UNK>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp

  cp data/local/dict_nosp/lexicon.txt data/lang_nosp/

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  # Build G
  local/arpa2fst.py data/local/lm/lm_tgmed.arpa |
    local/sym2int.pl -f 3 data/lang_nosp/words.txt >data/lang_nosp/G.fsa.txt

  echo "To load G:"
  echo "    Gfsa = k2.Fsa.from_openfst(<string of data/lang_nosp/G.fsa.txt>, acceptor=True)"
fi

if [ $stage -le 5 ]; then
  python3 ./prepare.py
fi

if [ $stage -le 6 ]; then
  python3 ./train.py
fi

if [ $stage -le 7 ]; then
  python3 ./decode.py
fi
