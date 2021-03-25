#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (author: Liyong Guo)
# Apache 2.0

import os
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path

# librispeech-lm-norm.txt is 4G
# train_960_text is 48M, which is stands for the sum of {train_clean_360, train_clean_100, train_other_500}
# here only train_960_text used to verify the whole pipeline
# A copy of train_960_text: "htts://drive.google.com/file/d/1AgP4wTqbfp12dv4fJmjKXHdOf8eOtp_A/view?usp=sharing"
# local_path: "/ceph-ly/open-source/snowfall/egs/librispeech/asr/simple_v1/data/local/lm_train/train_960_text"


def download_librispeech_train_960_text():
    train_960_text = "./data/lm_train/librispeech_train_960_text"
    if not os.path.exists(train_960_text):
        Path(os.path.dirname(train_960_text)).mkdir(parents=True,
                                                    exist_ok=True)

        logging.info("downloading train_960_text of librispeech.")
        gdd.download_file_from_google_drive(
            file_id='1AgP4wTqbfp12dv4fJmjKXHdOf8eOtp_A',
            dest_path=train_960_text,
            unzip=False)
    else:
        logging.info(
            "train_960_text of librispeech is already downloaded. You may should check that"
        )


def main():
    logging.getLogger().setLevel(logging.INFO)

    download_librispeech_train_960_text()


if __name__ == '__main__':
    main()
