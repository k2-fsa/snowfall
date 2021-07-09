#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

stage=0

libri_dirs=(
/root/fangjun/data/librispeech/LibriSpeech
/export/corpora5/LibriSpeech
/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech
/export/common/data/corpora/ASR/openslr/SLR12/LibriSpeech
)

libri_dir=
for d in ${libri_dirs[@]}; do
  if [ -d $d ]; then
    libri_dir=$d
    break
  fi
done

if [ ! -d $libri_dir/train-clean-100 ]; then
  echo "Please set LibriSpeech dataset path before running this script"
  exit 1
fi

echo "LibriSpeech dataset dir: $libri_dir"


if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
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

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  # Build G
  if [ ! -f data/lang_nosp/G_uni.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=1 \
      data/local/lm/lm_tgmed.arpa >data/lang_nosp/G_uni.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_uni.fst.txt"
  fi

  if [ ! -f data/lang_nosp/G.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/local/lm/lm_tgmed.arpa >data/lang_nosp/G.fst.txt
  else
    echo "Skip generating data/lang_nosp/G.fst.txt"
  fi

  if [ ! -f data/lang_nosp/G_4_gram.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/local/lm/lm_fglarge.arpa >data/lang_nosp/G_4_gram.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_4_gram.fst.txt"
  fi

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi

if [ $stage -le 5 ]; then
  mkdir -p data/local/tmp
  if [ ! -f data/local/tmp/transcript.txt ]; then
    echo "Generating data/local/tmp/transcript.txt"
    files=$(
      find "$libri_dir/train-clean-100" -name "*.trans.txt"
      find "$libri_dir/train-clean-360" -name "*.trans.txt"
      find "$libri_dir/train-other-500" -name "*.trans.txt"
    )
    for f in ${files[@]}; do
      cat $f | cut -d " " -f 2-
    done > data/local/tmp/transcript.txt
  fi
fi

if [ $stage -le 6 ]; then
  # this stage takes about 3 minutes
  mkdir -p data/lm
  if [ ! -f data/lm/P.arpa ]; then
    echo "Generating data/lm/P.arpa"
    ./local/add_silence_to_transcript.py \
      --transcript data/local/tmp/transcript.txt \
      --sil-word "!SIL" \
      --sil-prob 0.5 \
      --seed 20210629 \
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

if [ $stage -le 7 ]; then
  if [ ! -f data/lang_nosp/P.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/phones.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      data/lm/P.arpa > data/lang_nosp/P.fst.txt
  fi
fi

if [ $stage -le 8 ]; then
  python3 ./prepare.py
fi


# Normally, you should stop here and run the training script manually.
# stage 1 to 5 need only to be run once.
#
# exit 0

if [ $stage -le 9 ]; then
  # python3 ./train.py # ctc training
  # python3 ./mmi_bigram_train.py # ctc training + bigram phone LM
  # python3 ./mmi_mbr_train.py

  # python3 ./mmi_att_transformer_train.py --help

  # To use multi-gpu training, use
  # export CUDA_VISIBLE_DEVICES="0,1"
  # python3 ./mmi_att_transformer_train.py --world-size=2

  # single gpu training
  python3 ./mmi_att_transformer_train.py

  # Single node, multi-GPU training
  # Adapting to a multi-node scenario should be straightforward.
  # ngpus=2
  # python3 -m torch.distributed.launch --nproc_per_node=$ngpus ./mmi_bigram_train.py --world_size $ngpus
fi

if [ $stage -le 10 ]; then
  # python3 ./decode.py # ctc decoding
  # python3 ./mmi_bigram_decode.py --epoch 9
  #  python3 ./mmi_mbr_decode.py
  python3 ./mmi_att_transformer_decode.py
fi
