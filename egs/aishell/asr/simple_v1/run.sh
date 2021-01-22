#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

data=/export/data/asr-data/OpenSLR/33
stage=3

[ -f path.sh ] && . ./path.sh

if [ $stage -le 1 ]; then
  local2/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript
fi

if [ $stage -le 2 ]; then
  local2/aishell_prepare_dict.sh $data/resource_aishell data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  local/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
    "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  local2/aishell_train_lms.sh
  gunzip -c data/local/lm/3gram-mincount/lm_unpruned.gz > data/local/lm/3gram-mincount.arpa
  # Build G
  local/arpa2fst.py data/local/lm/3gram-mincount.arpa |
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
