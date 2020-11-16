#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Junbo Zhang, Haowen Qiu)
# Apache 2.0

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import torch
from lhotse import CutSet, Fbank, LilcomFilesWriter, WavAugmenter, RecordingSet, SupervisionSet
from lhotse.recipes.librispeech import dataset_parts_full, prepare_librispeech


# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() because it needs to take effect
# even when we are not invoking the main (notice: "spawn" is the method used
# in multiprocessing, which is to get around some problems with torchaudio's invocation of
# sox).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def main():
  print("All dataset parts: ", dataset_parts_full)

  dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100')
  print("Parts we will prepare: ", dataset_parts)
  assert(isinstance(dataset_parts[0], str) and len(dataset_parts[0]) > 1)

  corpus_dir = '/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech'
  output_dir = 'exp/data'
  try:
    librispeech_manifests = dict()
    for partition in dataset_parts:
      librispeech_manifests[partition] = dict()
      librispeech_manifests[partition]['recordings'] = RecordingSet.from_json(
        output_dir + f'/recordings_{partition}.json')
      librispeech_manifests[partition]['supervisions'] = SupervisionSet.from_json(
        output_dir + f'/supervisions_{partition}.json')
  except Exception as e:
    print("Librispeech manifests not found on disk, preparing them from scratch: ", str(e))
    librispeech_manifests = prepare_librispeech(corpus_dir, dataset_parts,
                                                output_dir)

  use_data_augmentation = False
  augmenter = WavAugmenter.create_predefined(
      'pitch_reverb_tdrop',
      sampling_rate=16000) if use_data_augmentation else None


  num_jobs = min(15, os.cpu_count())

  for partition, manifests in librispeech_manifests.items():
      print(partition)
      cut_set = CutSet.from_manifests(
        recordings=manifests['recordings'],
        supervisions=manifests['supervisions'])
      librispeech_manifests[partition]['cuts'] = cut_set
      cut_set.to_json(output_dir + f'/cuts_{partition}.json.gz')

      with LilcomFilesWriter(f'{output_dir}/feats_{partition}') as storage, \
             ProcessPoolExecutor(num_jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
          cut_set = cut_set.compute_and_store_features(
              extractor=Fbank(),
              executor=ex,
              storage=storage)

          if 'train' in partition:
              # Duplicate the training set with an augmented version
              augmented_cs = cut_set.compute_and_store_features(
                  extractor=Fbank(),
                  storage=storage,
                  executor=ex,
                  augment_fn=augmenter)
              cut_set = cut_set + CutSet.from_cuts(c.with_id(c.id + '_aug') for c in augmented_cs)
      librispeech_manifests[partition]['cuts'] = cut_set
      cut_set.to_json(output_dir + f'/cuts_{partition}.json.gz')


if __name__ == '__main__':
    main()
