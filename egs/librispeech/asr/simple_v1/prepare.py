#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Junbo Zhang, Haowen Qiu)
# Apache 2.0
import multiprocessing
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, LilcomFilesWriter, combine
from lhotse.augmentation import SoxEffectTransform
from lhotse.recipes.librispeech import prepare_librispeech

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() because it needs to take effect
# even when we are not invoking the main (notice: "spawn" is the method used
# in multiprocessing, which is to get around some problems with torchaudio's invocation of
# sox).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
num_jobs = min(15, os.cpu_count())


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output('hostname -f', shell=True, text=True)
        if name.strip().endswith('.clsp.jhu.edu'):
            import plz
            from distributed import Client
            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except:
        pass
    # Return a local node process pool (single node processing)
    yield ProcessPoolExecutor(num_jobs, mp_context=multiprocessing.get_context("spawn"))


def main():
    dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100')
    print("Parts we will prepare: ", dataset_parts)

    corpus_dir = Path('/export/corpora5/LibriSpeech')
    # corpus_dir = Path('/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech')
    output_dir = Path('exp/data')
    print('Manifest preparation:')
    librispeech_manifests = prepare_librispeech(
        corpus_dir=corpus_dir,
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        num_jobs=num_jobs
    )

    print('Feature extraction:')
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in librispeech_manifests.items():
            if (output_dir / f'cuts_{partition}.json.gz').is_file():
                print(f'{partition} already exists - skipping.')
                continue
            print('Processing', partition)

            cut_set = CutSet.from_manifests(
                recordings=manifests['recordings'],
                supervisions=manifests['supervisions']
            )

            cut_sets = []
            storage = LilcomFilesWriter(f'{output_dir}/feats_{partition}')
            cut_sets.append(
                cut_set.compute_and_store_features(
                    extractor=Fbank(),
                    executor=ex,
                    storage=storage
                )
            )
            if 'train' in partition:
                # Duplicate the training set with an augmented version
                # TODO(pzelasko): normally we would have used [0.9, 1.1] as in Kaldi,
                #   but I found out that we are cutting out too much speech in Lhotse
                #   if we slow down the audio. Change this once we fix it.
                #   (issue: https://github.com/lhotse-speech/lhotse/issues/166)
                for speed in [1.05, 1.1]:
                    cut_sets.append(
                        cut_set.modify_ids(
                            lambda cut_id: f'{cut_id}_sp{speed}'
                        ).compute_and_store_features(
                            extractor=Fbank(),
                            storage=storage,
                            executor=ex,
                            augment_fn=SoxEffectTransform([
                                ['speed', speed],
                                ['rate', 16000]
                            ])
                        )
                    )
            cut_set = combine(cut_sets)
            librispeech_manifests[partition]['cuts'] = cut_set
            cut_set.to_json(output_dir / f'cuts_{partition}.json.gz')


if __name__ == '__main__':
    main()
