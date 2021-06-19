#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

set -eou pipefail

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

stage=0

# settings for BPE training -- start
vocab_size=200
model_type=unigram # valid values: unigram, bpe, word, char
# settings for BPE training -- end

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
  echo "Preparing BPE training"
  dir=data/lang_bpe
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
    python3 ./train_bpe_model.py \
      --transcript $dir/transcript.txt \
      --model-type $model_type \
      --vocab-size $vocab_size \
      --output-dir $dir
  else
    echo "$model_file exists, skip BPE training"
  fi

  token_file=$dir/tokens.txt
  python3 ./generate_bpe_tokens.py \
    --model-file $model_file > $token_file
  # Copy tokens.txt to phones.txt since the existing code
  # expects a fixed name "phones.txt"
  cp $token_file $dir/phones.txt

  echo "<eps> 0" > $dir/words.txt
  echo "<UNK> 1" >> $dir/words.txt
  cat $dir/transcript.txt | tr -s " " "\n" | sort | uniq |
    awk '{print $0 " " NR+1}' >> $dir/words.txt

  if [ ! -f $dir/lexicon.txt ]; then
    python3 ./generate_bpe_lexicon.py \
      --model-file $model_file \
      --words-file $dir/words.txt > $dir/lexicon.txt
  fi

  if [ ! -f $dir/lexiconp.txt ]; then
    echo "**Creating $dir/lexiconp.txt from $dir/lexicon.txt"
    perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $dir/lexicon.txt > $dir/lexiconp.txt || exit 1
  fi

  # NOTE: 1 is <unk> in `--map-oov 1`.
  local/make_lexicon_fst.py $dir/lexiconp.txt | \
    local/sym2int.pl --map-oov 1 -f 3 $dir/tokens.txt | \
    local/sym2int.pl -f 4 $dir/words.txt > $dir/L.fst.txt || exit 1
fi

if [ $stage -le 6 ]; then
  python3 ./prepare.py
fi


# Normally, you should stop here and run the training script manually.
# stage 1 to 5 need only to be run once.
#
# exit 0

if [ $stage -le 7 ]; then
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

if [ $stage -le 8 ]; then
  # python3 ./decode.py # ctc decoding
  # python3 ./mmi_bigram_decode.py --epoch 9
  #  python3 ./mmi_mbr_decode.py
  python3 ./mmi_att_transformer_decode.py
fi
