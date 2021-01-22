#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Junbo Zhang, Haowen Qiu)
# Apache 2.0
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, LilcomHdf5Writer, combine
from lhotse.recipes import prepare_librispeech, prepare_musan

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
num_jobs = min(15, os.cpu_count())
full_libri = False


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
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


def main():
    if full_libri:
        dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100', 'train-clean-360', 'train-other-500')
    else:
        dataset_parts = ('dev-clean', 'test-clean', 'train-clean-100')

    print("Parts we will prepare: ", dataset_parts)

    corpus_dirs = [Path('/export/corpora5/LibriSpeech'),
                   Path('/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech')]
    corpus_dir = None
    for d in corpus_dirs:
        if os.path.exists(d):
            corpus_dir = d
    if corpus_dir is None:
        print("Please create a place on your system to put the downloaded Librispeech data "
              "and add it to `corpus_dirs`")
        sys.exit(1)

    output_dir = Path('exp/data')
    print('LibriSpeech manifest preparation:')
    librispeech_manifests = prepare_librispeech(
        corpus_dir=corpus_dir,
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        num_jobs=num_jobs
    )

    print('Musan manifest preparation:')
    musan_cuts_path = output_dir / 'cuts_musan.json.gz'
    musan_manifests = prepare_musan(
        corpus_dir='/export/corpora5/JHU/musan',
        output_dir=output_dir,
        parts=('music', 'speech', 'noise')
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
            if 'train' in partition:
                cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
            cut_set = cut_set.compute_and_store_features(
                extractor=Fbank(),
                storage_path=f'{output_dir}/feats_{partition}',
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer
            )
            librispeech_manifests[partition]['cuts'] = cut_set
            cut_set.to_json(output_dir / f'cuts_{partition}.json.gz')
        # Now onto Musan
        if not musan_cuts_path.is_file():
            print('Extracting features for Musan')
            # create chunks of Musan with duration 5 - 10 seconds
            musan_cuts = CutSet.from_manifests(
                recordings=combine(part['recordings'] for part in musan_manifests.values())
            ).cut_into_windows(10.0).filter(lambda c: c.duration > 5).compute_and_store_features(
                extractor=Fbank(),
                storage_path=f'{output_dir}/feats_musan',
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer
            )
            musan_cuts.to_json(musan_cuts_path)


if __name__ == '__main__':
    main()
