#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Junbo Zhang, Haowen Qiu)
# Apache 2.0

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import torch
from lhotse import CutSet, Fbank, LilcomFilesWriter, WavAugmenter
from lhotse.recipes.librispeech import dataset_parts_full, prepare_librispeech


def main():
  print("All dataset parts: ", dataset_parts_full)

  dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100')

  print("Parts we will prepare: ", dataset_parts)

  corpus_dir = '/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech'
  output_dir = 'exp/data'
  librispeech_manifests = prepare_librispeech(corpus_dir, dataset_parts,
                                              output_dir)

  use_data_augmentation = False
  augmenter = WavAugmenter.create_predefined(
      'pitch_reverb_tdrop',
      sampling_rate=16000) if use_data_augmentation else None
  # It seems when spawning multiple Python subprocesses with the same sox handle it raises "std::runtime_error: Couldn't close file"
  # The issue seems to happen only in a Jupyter notebook on Mac OS X, hence the work around below.
  if use_data_augmentation:
      num_jobs = 1
  else:
      num_jobs = os.cpu_count()
      torch.set_num_threads(1)
      torch.set_num_interop_threads(1)

  for partition, manifests in librispeech_manifests.items():
      print(partition)
      with LilcomFilesWriter(f'{output_dir}/feats_{partition}') as storage, \
              ProcessPoolExecutor(num_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
          cut_set = CutSet.from_manifests(
              recordings=manifests['recordings'],
              supervisions=manifests['supervisions'])
          cut_set = cut_set.compute_and_store_features(
              extractor=Fbank(),
              storage=storage,
              executor=ex)
          if 'train' in partition:
              # Duplicate the training set with an augmented version
              augmented_cs = cut_set.compute_and_store_features(
                  extractor=Fbank(),
                  storage=storage,
                  augment_fn=augmenter,
                  executor=ex)
              cut_set = cut_set + CutSet.from_cuts(c.with_id(c.id + '_aug') for c in augmented_cs)
      librispeech_manifests[partition]['cuts'] = cut_set
      cut_set.to_json(output_dir + f'/cuts_{partition}.json.gz')


if __name__ == '__main__':
    main()

