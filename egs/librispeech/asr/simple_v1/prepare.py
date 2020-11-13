#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from lhotse import CutSet, Fbank, Mfcc, LilcomFilesWriter, WavAugmenter
from lhotse.dataset import SpeechRecognitionDataset
from lhotse.recipes.librispeech import download_and_untar, prepare_librispeech, dataset_parts_full

print("All dataset parts: ", dataset_parts_full)

dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100')

print("Parts we will prepare: ", dataset_parts)

corpus_dir = '/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech'
output_dir = 'exp/data1'
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

num_jobs = 1
for partition, manifests in librispeech_manifests.items():
    print(partition)
    with LilcomFilesWriter(f'{output_dir}/feats_{partition}'
                           ) as storage, ProcessPoolExecutor(num_jobs) as ex:
        cut_set = CutSet.from_manifests(
            recordings=manifests['recordings'],
            supervisions=manifests['supervisions']).compute_and_store_features(
                extractor=Fbank(),
                storage=storage,
                augment_fn=augmenter if 'train' in partition else None,
                executor=ex)
    librispeech_manifests[partition]['cuts'] = cut_set
    cut_set.to_json(output_dir + f'/cuts_{partition}.json.gz')

cuts_train = SpeechRecognitionDataset(
    librispeech_manifests['train-clean-100']['cuts'])
cuts_test = SpeechRecognitionDataset(
    librispeech_manifests['test-clean']['cuts'])

sample = cuts_train[0]
print('Transcript:', sample['text'])
print('Supervisions mask:', sample['supervisions_mask'])
print('Feature matrix:', sample.load_features())
