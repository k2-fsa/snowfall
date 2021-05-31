#!/usr/bin/env bash

# Copyright 2020-2021 Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

stage=0

# GigaSpeech configurations.
gigaspeech_root=~/data/giga
gigaspeech_train_subset=XL
gigaspeech_test_sets="gigaspeech_dev gigaspeech_test"
gigaspeech_train_sets="gigaspeech_train_${gigaspeech_train_subset,,}"

# G2P models.
g2p_model=$gigaspeech_root/dict/g2p/g2p.model.4

# Experiment configurations.
lm_order=3
lm_dir=data/local/lm
dict_dir=data/local/dict

# Train/Dev/Test sets.
test_sets="$gigaspeech_test_sets"
train_sets="$gigaspeech_train_sets"
train_combined="$gigaspeech_train_sets"

. local/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
  local/data_prep.sh --train-subset $gigaspeech_train_subset $gigaspeech_root data || exit 1;
fi

if [ $stage -le 1 ]; then
  [ ! -f $g2p_model ] && echo "$0: Cannot find G2P model $g2p_model" && exit 1
  local/prepare_dict.sh --nj 50 $g2p_model data/$train_combined $dict_dir || exit 1;
fi

if [ $stage -le 3 ]; then
  mkdir -p $lm_dir || exit 1;
  sed 's|\t| |' data/$train_combined/text |\
    cut -d " " -f 2- > $lm_dir/corpus.txt || exit 1;
  local/train_lm.sh --lm-order $lm_order $lm_dir/corpus.txt $lm_dir || exit 1;
fi

if [ $stage -le 4 ]; then
  local/prepare_lang.sh $dict_dir \
    "<UNK>" data/local/lang_tmp data/lang_nosp || exit 1;

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 5 ]; then
  # Build G
  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=1 \
    data/local/lm/lm_3gram.arpa >data/lang_nosp/G_uni.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/local/lm/lm_3gram.arpa >data/lang_nosp/G.fst.txt

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi

if [ $stage -le 6 ]; then
  python3 ./prepare.py
fi

if [ $stage -le 7 ]; then
  ngpus=2
  python3 -m torch.distributed.launch --nproc_per_node=$ngpus ./mmi_bigram_train.py --world_size $ngpus
fi

if [ $stage -le 8 ]; then
  python3 ./mmi_bigram_decode.py --epoch 9
fi
