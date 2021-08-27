#!/usr/bin/env bash

# Copyright 2017 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  echo " $0 /export/a05/xna/data/data_aishell/wav /export/a05/xna/data/data_aishell/transcript"
  exit 1;
fi

aishell_audio_dir=$1
aishell_text=$2/aishell_transcript_v0.8.txt

train_dir=data/local/train
tmp_dir=data/local/tmp

mkdir -p $train_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -f $aishell_text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && \
  echo Warning: expected 141925 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  local/filter_scp.pl -f 1 $dir/utt.list $aishell_text > $dir/transcripts.txt
  sort -u $dir/transcripts.txt > $dir/text
done


echo "$0: AISHELL data preparation succeeded"
exit 0;
