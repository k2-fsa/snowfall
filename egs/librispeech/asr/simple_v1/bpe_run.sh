#!/usr/bin/env bash

# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo)
# Apache 2.0

# Example of Bpe training with ctc loss and lable smooth loss

stage=0
download_model=1
if [ $download_model -eq 1 ]; then
  echo "Use pretrained model from model zoo for decoding and evaluation"
  if [ -d snowfall_bpe_model ]; then
    echo "Model seems already been dowloaded"
  else
    if ! type git-lfs >/dev/null 2>&1; then
          echo 'Please Install git-lfs to download trained models';
          exit 0
    fi
    git clone https://huggingface.co/GuoLiyong/snowfall_bpe_model
    for sub_dir in data exp-bpe-lrfactor10.0-conformer-512-8-noam; do
      ln -sf snowfall_bpe_model/$sub_dir ./
    done
  fi
fi

if [ ! -f exp/data/cuts_test-clean.json.gz ]; then
  echo "Please follow stage 5 in ./run.sh to generate test manifests"
  exit 1
fi

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  dir=data/lang_bpe2
  mkdir -p $dir
  token_file=./data/en_token_list/bpe_unigram5000/tokens.txt
  model_file=./data/en_token_list/bpe_unigram5000/bpe.model
  if [ ! -f $dir/tokens.txt ]; then
    cp $token_file $dir/tokens.txt
    ln -fv $dir/tokens.txt $dir/phones.txt
    echo "<eps> 0" > $dir/words.txt
    echo "<UNK> 1" >> $dir/words.txt
    cat data/local/lm/librispeech-vocab.txt | sort | uniq |
      awk '{print $1 " " NR+1}' >> $dir/words.txt
  fi

  if [ ! -f $dir/lexicon.txt ]; then
    python3 ./generate_bpe_lexicon.py \
      --model-file $model_file \
      --words-file $dir/words.txt > $dir/lexicon.txt
  fi

  if [ ! -f $dir/lexiconp.txt ]; then
    echo "**Creating $dir/lexiconp.txt from $dir/lexicon.txt"
    perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $dir/lexicon.txt > $dir/lexiconp.txt || exit 1
  fi

  if ! grep "#0" $dir/words.txt > /dev/null 2>&1; then
    max_word_id=$(tail -1 $dir/words.txt | awk '{print $2}')
    echo "#0 $((max_word_id+1))" >> $dir/words.txt
  fi

  ndisambig=$(local/add_lex_disambig.pl --pron-probs $dir/lexiconp.txt $dir/lexiconp_disambig.txt)
  if ! grep "#0" $dir/phones.txt > /dev/null 2>&1 ; then
    max_phone_id=$(tail -1 $dir/phones.txt | awk '{print $2}')
    for i in $(seq 0 $ndisambig); do
      echo "#$i $((i+max_phone_id+1))"
    done >> $dir/phones.txt
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
  if [ ! -f $dir/G_4_gram.fst.txt ]; then
    python3 -m kaldilm \
      --read-symbol-table="$dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/local/lm/lm_fglarge.arpa >$dir/G_4_gram.fst.txt
  else
    echo "Skip generating data/lang_nosp/G_4_gram.fst.txt"
  fi
fi

if [ $stage -le 3 ]; then
  export CUDA_VISIBLE_DEVICES=2
  python bpe_ctc_att_conformer_decode.py \
    --max-duration=20 \
    --generate-release-model=False \
    --decode_with_released_model=True \
    --num-paths-for-decoder-rescore=500
fi
