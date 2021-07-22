#!/usr/bin/env bash

# Copyright 2021 Johns Hopkins University (author: Piotr Żelasko)
#           2021 Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

set -eou pipefail

stage=0
subset='XS'
nj=50

gigaspeech_dirs=(
/export/corpora5/gigaspeech
/exp/swatanabe/data/gigaspeech/
/home/storage07/zhangjunbo/data/gigaspeech
)

giga_dir=
for d in ${gigaspeech_dirs[@]}; do
  if [ -d $d ]; then
    giga_dir=$d
    break
  fi
done

gigaspeech_test_sets="gigaspeech_dev gigaspeech_test"
gigaspeech_train_sets="gigaspeech_train_${subset,,}"

# G2P and LM configure.
g2p_model=$giga_dir/dict/g2p/g2p.model.4
lm_order=3
lm_dir=data/local/lm
dict_dir=data/local/dict

if [ ! -f $giga_dir/GigaSpeech.json ]; then
  echo "Please set GigaSpeech dataset path before running this script"
  exit 1
fi

echo "GigaSpeech dataset dir: $giga_dir"


if [ ! -d GigaSpeech ]; then
  git clone https://github.com/SpeechColab/GigaSpeech
fi

if [ $stage -le 0 ]; then
  local/data_prep.sh --train-subset $subset $giga_dir data || exit 1;
fi

if [ $stage -le 1 ]; then
  [ ! -f $g2p_model ] && echo "$0: Cannot find G2P model $g2p_model" && exit 1
  local/prepare_dict.sh --nj 50 $g2p_model data/$gigaspeech_train_sets $dict_dir || exit 1;
fi

if [ $stage -le 3 ]; then
  mkdir -p $lm_dir || exit 1;
  sed 's|\t| |' data/$train_combined/text |\
    cut -d " " -f 2- > $lm_dir/corpus.txt || exit 1;
  local/train_lm.sh --lm-order $lm_order $lm_dir/corpus.txt $lm_dir || exit 1;
fi

if [ $stage -le 4 ]; then
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_nosp \
    "<UNK>" \
    data/local/lang_tmp_nosp \
    data/lang_nosp
fi

if [ $stage -le 5 ]; then
  # Build G
#  if [ ! -f data/lang_nosp/G_uni.fst.txt ]; then
#    python3 -m kaldilm \
#      --read-symbol-table="data/lang_nosp/words.txt" \
#      --disambig-symbol='#0' \
#      --max-order=1 \
#      data/local/lm/lm_tgmed.arpa >data/lang_nosp/G_uni.fst.txt
#  else
#    echo "Skip generating data/lang_nosp/G_uni.fst.txt"
#  fi

  if [ ! -f data/lang_nosp/G.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/local/lm/lm_4gram.arpa >data/lang_nosp/G.fst.txt
  else
    echo "Skip generating data/lang_nosp/G.fst.txt"
  fi

  if [ ! -f data/lang_nosp/G_4_gram.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/local/lm/lm_4gram.arpa >data/lang_nosp/G_4_gram.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_4_gram.fst.txt"
  fi
fi

if [ $stage -le 6 ]; then
  python3 ./prepare.py --subset $subset
fi

if [ $stage -le 7 ]; then
  mkdir -p data/local/tmp
  if [ ! -f data/local/tmp/transcript.txt ]; then
    echo "Generating data/local/tmp/transcript.txt"
    # extract text field \
    # | remove quotes \
    # > save
    jq '.text' "exp/data/supervisions_{${subset}}.jsonl" \
     | sed 's/"//g' \
     > data/local/tmp/transcript.txt
  fi
fi

if [ $stage -le 8 ]; then
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

if [ $stage -le 9 ]; then
  if [ ! -f data/lang_nosp/P.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="data/lang_nosp/phones.txt" \
      --disambig-symbol='#0' \
      --max-order=2 \
      data/lm/P.arpa > data/lang_nosp/P.fst.txt
  fi
fi


# Normally, you should stop here and run the training script manually.
# stage 1 to 5 need only to be run once.
#
# exit 0

if [ $stage -le 10 ]; then
  python3 ./mmi_att_transformer_train.py --subset "$subset"
fi

if [ $stage -le 11 ]; then
  python3 ./mmi_att_transformer_decode.py
fi
