  
#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Mingshuang Luo)
# Apache 2.0

from collections import defaultdict

import os
import glob
import json
import logging
import shutil
import tarfile
import string
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Optional, Union
from concurrent.futures.thread import ThreadPoolExecutor

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress


def download_and_unzip(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        base_url: Optional[str] = 'https://data.deepai.org/timit.zip') -> None:
    """
    Download and unzip the dataset, supporting both TIMIT

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the zips no matter if the zips exists.
    :param base_url: str, the url of the TIMIT download for free.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_name = f'timit.zip'
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urlretrieve_progress(f'{base_url}', filename=tar_path, desc=f'Downloading {tar_name}')
    

def prepare_timit(
        corpus_dir: Pathlike,
        output_dir: Optional[Pathlike] = None,
        num_jobs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    dataset_parts = ['TRAIN', 'DEV', 'TEST']
    
    punctuation_strings = string.punctuation
    
    id_texts = []
    
    with ThreadPoolExecutor(num_jobs) as ex:
        for part in dataset_parts:
          wav_files = []
          file_name = ''
          pwd_dir = os.getcwd()
          if part == 'TRAIN':
            file_name = os.path.join(pwd_dir, 'local1/train_samples.txt') 
          elif part == 'DEV':
            file_name = os.path.join(pwd_dir, 'local1/dev_samples.txt')
          else:
            file_name = os.path.join(pwd_dir, 'tst_samples.txt')
          wav_files = []
          with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(' ')
                wav = os.path.join(corpus_dir, items[-1])
                wav_files.append(wav)
            print(f'{part} dataset manifest generation.')
            recordings = []
            supervisions = []
            
            for wav_file in tqdm(wav_files):
                items = wav_file.split('/')
                idx = items[-2] + '-' + items[-1][:-4]
                speaker = items[-2]
                transcript_file = wav_file[:-3] + 'PHN' ###the phone file
                if not Path(wav_file).is_file():
                    logging.warning(f'No such file: {wav_file}')
                    continue
                if not Path(transcript_file).is_file():
                    logging.warning(f'No transcript: {transcript_file}')
                    continue
                text = []
                with open(transcript_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        phone = line.rstrip('\n').split(' ')[-1]
                        text.append(phone)
                    text = ' '.join(text).replace('h#', 'sil')
                    
                for i in punctuation_strings:
                    if i != "'":
                        text = text.replace(i, '')
                
                recording = Recording.from_file(path=wav_file, recording_id=idx)
                recordings.append(recording)
                segment = SupervisionSegment(
                    id=idx,
                    recording_id=idx,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language='English',
                    speaker=speaker,
                    text=text.strip())

                supervisions.append(segment)

                recording_set = RecordingSet.from_recordings(recordings)
                supervision_set = SupervisionSet.from_segments(supervisions)
                validate_recordings_and_supervisions(recording_set, supervision_set)

                if output_dir is not None:
                    supervision_set.to_json(output_dir / f'supervisions_{part}.json')
                    recording_set.to_json(output_dir / f'recordings_{part}.json')

                manifests[part] = {
                    'recordings': recording_set,
                    'supervisions': supervision_set}

    return manifests

'''
if __name__ == "__main__":
    corpus_dir = Path('/kome/luomingshuang/audio-data/timit')
    #download_and_unzip(corpus_dir, force_download=True)
    output_dir = Path('/kome/luomingshuang/snowfall/egs/librispeech2timit/asr/simple_v1/exp1/data')
    timit_manifests = prepare_timit(corpus_dir=corpus_dir, output_dir=output_dir, num_jobs=8)
'''
