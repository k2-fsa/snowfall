#!/usr/bin/env bash

# Copyright 2021 Xiaomi Corporation (Authors: Fangjun Kuang
#                                             Wei Kang)
# Apache 2.0

set -eou pipefail

libri_dirs=(
/kome/kangwei/data/librispeech/LibriSpeech
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

stage=4


if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  # settings for BPE training -- start
  vocab_size=5000
  model_type=unigram # valid values: unigram, bpe, word, char
  # settings for BPE training -- end

  echo "Preparing BPE training"
  dir=data/lang_bpe
  mkdir -p $dir
  if [ ! -f $dir/transcript.txt ]; then
    echo "Generating $dir/transcript.txt"
    files=$(
      find "$libri_dir/train-clean-100" -name "*.trans.txt"
      find "$libri_dir/train-clean-360" -name "*.trans.txt"
      find "$libri_dir/train-other-500" -name "*.trans.txt"
    )
    for f in ${files[@]}; do
      cat $f | cut -d " " -f 2-
    done > $dir/transcript.txt
  fi

  model_file=$dir/bpe_${model_type}_${vocab_size}.model
  if [ ! -f $model_file ]; then
    echo "Generating $model_file"
    python3 ./local/train_bpe_model.py \
      --transcript $dir/transcript.txt \
      --model-type $model_type \
      --vocab-size $vocab_size \
      --output-dir $dir
  else
    echo "$model_file exists, skip BPE training"
  fi

  if [ ! -f $dir/tokens.txt ]; then
    python3 ./local/generate_bpe_tokens.py \
      --model-file $model_file > $dir/tokens.txt
  fi
  # Copy tokens.txt to phones.txt since the existing code
  # expects a fixed name "phones.txt"
  ln -fv $dir/tokens.txt $dir/phones.txt

  if [ ! -f $dir/words.txt ]; then
    echo "<eps> 0" > $dir/words.txt
    echo "<UNK> 1" >> $dir/words.txt
    cat $dir/transcript.txt | tr -s " " "\n" | sort | uniq |
      awk '{print $0 " " NR+1}' >> $dir/words.txt
  fi

  if [ ! -f $dir/lexicon.txt ]; then
    python3 ./local/generate_bpe_lexicon.py \
      --model-file $model_file \
      --words-file $dir/words.txt > $dir/lexicon.txt
  fi

  if [ ! -f $dir/lexiconp.txt ]; then
    echo "**Creating $dir/lexiconp.txt from $dir/lexicon.txt"
    perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $dir/lexicon.txt > $dir/lexiconp.txt || exit 1
  fi

  ndisambig=$(local/add_lex_disambig.pl --pron-probs $dir/lexiconp.txt $dir/lexiconp_disambig.txt)
  if ! grep "#0" $dir/words.txt > /dev/null 2>&1; then
    max_word_id=$(tail -1 $dir/words.txt | awk '{print $2}')
    for i in $(seq 0 $ndisambig); do
      echo "#$i $((i+max_word_id+1))"
    done >> $dir/words.txt
  fi

  if ! grep "#0" $dir/phones.txt > /dev/null 2>&1 ; then
    max_phone_id=$(tail -1 $dir/phones.txt | awk '{print $2}')
    for i in $(seq 0 $ndisambig); do
      echo "#$i $((i+max_phone_id+1))"
    done >> $dir/phones.txt
  fi

  if [ ! -f $dir/L.fst.txt ]; then
    # NOTE: 1 is <unk> in `--map-oov 1`.
    local/make_lexicon_fst.py $dir/lexiconp.txt | \
      local/sym2int.pl --map-oov 1 -f 3 $dir/tokens.txt | \
      local/sym2int.pl -f 4 $dir/words.txt > $dir/L.fst.txt || exit 1
  fi

  if [ ! -f $dir/L_disambig.fst.txt ]; then
    wdisambig_phone=$(echo "#0" | local/sym2int.pl $dir/phones.txt)
    wdisambig_word=$(echo "#0" | local/sym2int.pl $dir/words.txt)

    local/make_lexicon_fst.py \
      $dir/lexiconp_disambig.txt | \
      local/sym2int.pl --map-oov 1 -f 3 $dir/phones.txt | \
      local/sym2int.pl -f 4 $dir/words.txt | \
      local/fstaddselfloops.pl $wdisambig_phone $wdisambig_word > $dir/L_disambig.fst.txt || exit 1
  fi

  if [ ! -f $dir/G.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/local/lm/lm_tgmed.arpa > $dir/G.fst.txt
  else
    echo "Skip generating $dir/G.fst.txt"
  fi

#  if [ ! -f $dir/G_4_gram.fst.txt ]; then
#    python3 -m kaldilm \
#      --read-symbol-table="$dir/words.txt" \
#      --disambig-symbol='#0' \
#      --max-order=4 \
#      data/local/lm/lm_fglarge.arpa > $dir/G_4_gram.fst.txt
#  else
#    echo "Skip generating $dir/G_4_gram.fst.txt"
#  fi
fi
# exit 0

if [ $stage -le 3 ]; then
  python3 ./prepare.py
fi

# Normally, you should stop here and run the training script manually.
# stage 1 to 3 need only to be run once.

if [ $stage -le 4 ]; then
  python3 ./ctc_att_transformer_train.py --world-size 3 \
    --nhead 8 \
    --attention-dim 512 \
    --num-epochs 50 \
    --full-libri True \
    --max-duration 200
fi

if [ $stage -le 5 ]; then
  python3 ./ctc_att_transformer_decode.py \
    --nhead 8 \
    --attention-dim 512 \
    --max-duration 100 \
    --epoch 12
fi
