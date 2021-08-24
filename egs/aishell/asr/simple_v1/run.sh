#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang
#                                            Mingshuang Luo)
#           2021 Pingfeng Luo
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

dataset_path=(
  /ceph-meixu/luomingshuang/audio-data/aishell
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
  local2/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript
fi

if [ $stage -le 2 ]; then
  echo "Dict preparation"
  local2/aishell_prepare_dict.sh $data/resource_aishell data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  echo "Lang preparation"
  local/prepare_lang.sh --position-dependent-phones false data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp || exit 1

  echo "To load L:"
  echo "Use::"
  echo "  with open('data/lang_nosp/L.fst.txt') as f:"
  echo "    Lfst = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi

if [ $stage -le 4 ]; then
  echo "LM preparation"
  local2/aishell_train_lms.sh
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
  mkdir -p data/lm
  # this stage may takes some minutes
  if [ ! -f data/lm/P.arpa ]; then
    echo "Generating data/lm/P.arpa"
    ./local/add_silence_to_transcript.py \
      --transcript data/local/train/transcripts.txt \
      --sil-word "!SIL" \
      --sil-prob 0.5 \
      --seed 20210823 \
      > data/lm/transcript_with_sil.txt

    ./local/convert_transcript_to_corpus.py \
      --transcript data/lm/transcript_with_sil.txt \
      --lexicon data/local/dict_nosp/lexicon.txt \
      --oov "<UNK>" \
      > data/lm/corpus.txt

    ./local/make_kn_lm.py \
      -ngram-order 2 \
      -text data/lm/corpus.txt \
      -lm data/lm/P.arpa
  fi
fi

if [ $stage -le 6 ]; then
  if [ ! -f data/lang_nosp/P.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/phones.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      data/lm/P.arpa > data/lang_nosp/P.fst.txt
  fi
fi

if [ $stage -le 7 ]; then
  echo "Feature preparation"
  python3 ./prepare.py
fi

if [ $stage -le 8 ]; then
  echo "Training"
  python3 ./ctc_train.py
  #python3 ./mmi_bigram_train.py
  #python3 ./mmi_att_transformer_train.py
fi

if [ $stage -le 9 ]; then
  echo "Decoding"
  python3 ./ctc_decode.py
  #python3 ./mmi_bigram_decode.py
  #python3 ./mmi_att_transformer_decode.py
fi
