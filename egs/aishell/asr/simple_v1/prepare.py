#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Junbo Zhang, Haowen Qiu)
#                2021  Pingfeng Luo
# Apache 2.0
import multiprocessing
import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, LilcomHdf5Writer, combine
from lhotse.recipes import prepare_aishell, prepare_musan

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() because it needs to take effect
# even when we are not invoking the main (notice: "spawn" is the method used
# in multiprocessing, which is to get around some problems with torchaudio's invocation of
# sox).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
num_jobs = min(15, os.cpu_count())
print(num_jobs)

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
    if sys.version_info < (3, 7, 9):
        yield ProcessPoolExecutor(num_jobs)
    else:
        yield ProcessPoolExecutor(num_jobs, mp_context=multiprocessing.get_context('spawn'))


def locate_corpus(corpus_dirs, msg):
    for d in corpus_dirs:
        if os.path.exists(str(d)):
            return d
    print(msg)
    sys.exit(1)

def main():
    corpus_dir = locate_corpus(
        (Path('/mnt/cfs2/asr/database/AM/aishell'),
         Path('/root/fangjun/data/aishell'),
         Path(
             '/home/storage04/zhuangweiji/data/open-source-data/SLR33-aishell/data'
         )),
        msg='Please specify the directory to the AIShell dataset')

    musan_dir = locate_corpus(
        (Path('/export/corpora5/JHU/musan'),
         Path('/export/common/data/corpora/MUSAN/musan'),
         Path('/root/fangjun/data/musan')),
        msg='Please specify the directory to the MUSAN dataset')

    output_dir = Path('exp/data')
    print('aishell manifest preparation:')
    aishell_manifests = prepare_aishell(
        corpus_dir=corpus_dir,
        output_dir=output_dir
    )

    print('Musan manifest preparation:')
    musan_cuts_path = output_dir / 'cuts_musan.json.gz'
    musan_manifests = prepare_musan(
        corpus_dir=musan_dir,
        output_dir=output_dir,
        parts=('music', 'speech', 'noise')
    )

    print('Feature extraction:')
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in aishell_manifests.items():
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
                num_jobs=num_jobs if ex is not None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer
            )
            aishell_manifests[partition]['cuts'] = cut_set
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
