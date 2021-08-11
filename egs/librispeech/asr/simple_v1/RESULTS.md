# Results

# LibriSpeech CTC training results (TDNN-LSTM)

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/l3dVVLgBRIOqBY4u3e4dbA/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/5995756/log-train-2021-02-16-12-36-09.txt>.

Decoding results (WER) of each epoch are listed below. They are obtained using the latest k2 and lhotse as of today (
2021-02-17).

```
# epoch 0
2021-02-17 09:52:22,159 INFO [ctc_decode.py:176] %WER 18.86% [9914 / 52576, 1012 ins, 1317 del, 7585 sub ]

# epoch 1
2021-02-17 09:53:06,097 INFO [ctc_decode.py:176] %WER 16.05% [8441 / 52576, 949 ins, 1016 del, 6476 sub ]

# epoch 2
2021-02-17 09:53:48,266 INFO [ctc_decode.py:176] %WER 14.36% [7551 / 52576, 896 ins, 851 del, 5804 sub ]

# epoch 3
2021-02-17 09:54:31,108 INFO [ctc_decode.py:176] %WER 14.09% [7409 / 52576, 881 ins, 829 del, 5699 sub ]

# epoch 4
2021-02-17 09:55:12,404 INFO [ctc_decode.py:176] %WER 13.84% [7274 / 52576, 871 ins, 781 del, 5622 sub ]

# epoch 5
2021-02-17 09:55:53,672 INFO [ctc_decode.py:176] %WER 13.47% [7080 / 52576, 893 ins, 708 del, 5479 sub ]

# epoch 6
2021-02-17 09:56:34,004 INFO [ctc_decode.py:176] %WER 13.01% [6840 / 52576, 862 ins, 707 del, 5271 sub ]

# epoch 7
2021-02-17 09:57:13,215 INFO [ctc_decode.py:176] %WER 12.91% [6787 / 52576, 797 ins, 792 del, 5198 sub ]
```

# LibriSpeech MMI+Bigram training results (TDNN-LSTM)

## 2021-08-11 (with BucketingSampler)

(Mingshuang Luo): Results of <https://github.com/k2-fsa/snowfall/pull/244>

TensorBoard log is available at <https://tensorboard.dev/experiment/ziBwrvVWQtmZgMFggvNQUA/#scalars>.

Decoding results (WER) of each epoch are listed below. They are obtained using the latest k2 and lhotse as of today (
2021-08-11).

```
# epoch 0
2021-08-11 13:24:20,663 INFO [common.py:414] [test-clean] %WER 14.95% [7860 / 52576, 840 ins, 969 del, 6051 sub ]

# epoch 1
2021-08-11 13:26:05,054 INFO [common.py:414] [test-clean] %WER 14.00% [7359 / 52576, 853 ins, 836 del, 5670 sub ]

# epoch 2
2021-08-11 13:36:05,913 INFO [common.py:414] [test-clean] %WER 12.29% [6459 / 52576, 835 ins, 607 del, 5017 sub ]

# epoch 3
2021-08-11 13:38:27,088 INFO [common.py:414] [test-clean] %WER 12.06% [6340 / 52576, 765 ins, 598 del, 4977 sub ]

# epoch 4
2021-08-11 13:41:20,207 INFO [common.py:414] [test-clean] %WER 11.42% [6003 / 52576, 679 ins, 622 del, 4702 sub ]

# epoch 5
2021-08-11 13:43:36,442 INFO [common.py:414] [test-clean] %WER 11.24% [5909 / 52576, 778 ins, 527 del, 4604 sub ]

# epoch 6
2021-08-11 13:45:12,678 INFO [common.py:414] [test-clean] %WER 11.34% [5961 / 52576, 706 ins, 635 del, 4620 sub ]

# epoch 7
2021-08-11 13:46:56,316 INFO [common.py:414] [test-clean] %WER 11.28% [5932 / 52576, 798 ins, 516 del, 4618 sub ]

# epoch 8
2021-08-11 13:49:38,579 INFO [common.py:414] [test-clean] %WER 10.72% [5638 / 52576, 747 ins, 524 del, 4367 sub ]

# epoch 9
2021-08-11 13:53:08,955 INFO [common.py:414] [test-clean] %WER 10.44% [5491 / 52576, 719 ins, 510 del, 4262 sub ]
```

# LibriSpeech MMI+MBR training results (TDNN-LSTM)

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/fYDCXr2iQWOwZQYQXTOZLQ/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/5995761/log-train-2021-02-17-01-10-23.txt>.

Decoding results (WER) of each epoch are listed below. They are obtained using the latest k2 and lhotse as of today (
2021-02-17).

```
# epoch 0
2021-02-17 11:47:42,262 INFO [mmi_mbr_decode.py:252] %WER 18.29% [9615 / 52576, 1211 ins, 1046 del, 7358 sub ]

# epoch 1
2021-02-17 11:48:28,176 INFO [mmi_mbr_decode.py:252] %WER 15.87% [8343 / 52576, 907 ins, 1024 del, 6412 sub ]

# epoch 2
2021-02-17 11:49:12,535 INFO [mmi_mbr_decode.py:252] %WER 15.36% [8076 / 52576, 858 ins, 1010 del, 6208 sub ]

# epoch 3
2021-02-17 11:49:59,331 INFO [mmi_mbr_decode.py:252] %WER 15.04% [7908 / 52576, 951 ins, 844 del, 6113 sub ]

# epoch 4
2021-02-17 11:50:46,515 INFO [mmi_mbr_decode.py:252] %WER 15.10% [7940 / 52576, 947 ins, 902 del, 6091 sub ]

# epoch 5
2021-02-17 11:51:33,309 INFO [mmi_mbr_decode.py:252] %WER 14.73% [7744 / 52576, 944 ins, 857 del, 5943 sub ]

# epoch 6
2021-02-17 11:52:20,559 INFO [mmi_mbr_decode.py:252] %WER 14.99% [7883 / 52576, 969 ins, 833 del, 6081 sub ]

# epoch 7
2021-02-17 11:53:05,260 INFO [mmi_mbr_decode.py:252] %WER 14.10% [7413 / 52576, 932 ins, 773 del, 5708 sub ]

# epoch 8
2021-02-17 11:53:50,381 INFO [mmi_mbr_decode.py:252] %WER 13.31% [6997 / 52576, 838 ins, 752 del, 5407 sub ]

# epoch 9
2021-02-17 11:54:34,531 INFO [mmi_mbr_decode.py:252] %WER 12.95% [6810 / 52576, 831 ins, 728 del, 5251 sub ]
```

# LibriSpeech CTC training results

## 2021-02-17 (Transformer)

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/SO24KcJ6RsONmPTjaNbHgQ/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/5995766/log-train-2021-02-16-02-27-50.txt>.

Decoding results (WER) of final model averaged over last 5 epochs and each epoch model without model averaging are
listed below. They are obtained using the latest k2 and lhotse as of today (2021-02-17).

```
# average over last 5 epochs
2021-02-17 08:41:19,214 INFO [ctc_att_transformer_decode.py:226] %WER 8.15% [4287 / 52576, 482 ins, 549 del, 3256 sub ]

# epoch 0
2021-02-17 08:26:21,801 INFO [ctc_att_transformer_decode.py:226] %WER 54.84% [28835 / 52576, 329 ins, 14178 del, 14328 sub ]

# epoch 1
2021-02-17 08:27:18,264 INFO [ctc_att_transformer_decode.py:226] %WER 38.53% [20260 / 52576, 368 ins, 8748 del, 11144 sub ]

# epoch 2
2021-02-17 08:28:06,537 INFO [ctc_att_transformer_decode.py:226] %WER 20.53% [10794 / 52576, 482 ins, 3225 del, 7087 sub ]

# epoch 3
2021-02-17 08:28:51,410 INFO [ctc_att_transformer_decode.py:226] %WER 14.96% [7866 / 52576, 501 ins, 1995 del, 5370 sub ]

# epoch 4
2021-02-17 08:29:33,906 INFO [ctc_att_transformer_decode.py:226] %WER 13.20% [6941 / 52576, 497 ins, 1523 del, 4921 sub ]

# epoch 5
2021-02-17 08:30:15,549 INFO [ctc_att_transformer_decode.py:226] %WER 11.99% [6304 / 52576, 564 ins, 1192 del, 4548 sub ]

# epoch 6
2021-02-17 08:30:55,740 INFO [ctc_att_transformer_decode.py:226] %WER 11.22% [5898 / 52576, 512 ins, 1073 del, 4313 sub ]

# epoch 7
2021-02-17 08:31:36,549 INFO [ctc_att_transformer_decode.py:226] %WER 11.37% [5980 / 52576, 461 ins, 1273 del, 4246 sub ]

# epoch 8
2021-02-17 08:32:16,219 INFO [ctc_att_transformer_decode.py:226] %WER 10.50% [5521 / 52576, 538 ins, 918 del, 4065 sub ]

# epoch 9
2021-02-17 08:32:55,890 INFO [ctc_att_transformer_decode.py:226] %WER 10.46% [5501 / 52576, 489 ins, 982 del, 4030 sub ]

# epoch 10
2021-02-17 08:33:35,887 INFO [ctc_att_transformer_decode.py:226] %WER 10.17% [5348 / 52576, 563 ins, 801 del, 3984 sub ]

# epoch 11
2021-02-17 08:34:16,014 INFO [ctc_att_transformer_decode.py:226] %WER 9.77% [5136 / 52576, 535 ins, 776 del, 3825 sub ]

# epoch 12
2021-02-17 08:34:55,230 INFO [ctc_att_transformer_decode.py:226] %WER 10.00% [5258 / 52576, 502 ins, 856 del, 3900 sub ]

# epoch 13
2021-02-17 08:35:34,472 INFO [ctc_att_transformer_decode.py:226] %WER 9.76% [5132 / 52576, 466 ins, 850 del, 3816 sub ]

# epoch 14
2021-02-17 08:36:12,854 INFO [ctc_att_transformer_decode.py:226] %WER 9.35% [4914 / 52576, 457 ins, 778 del, 3679 sub ]

# epoch 15
2021-02-17 08:36:50,909 INFO [ctc_att_transformer_decode.py:226] %WER 9.16% [4818 / 52576, 457 ins, 717 del, 3644 sub ]

# epoch 16
2021-02-17 08:37:28,806 INFO [ctc_att_transformer_decode.py:226] %WER 9.22% [4846 / 52576, 475 ins, 721 del, 3650 sub ]

# epoch 17
2021-02-17 08:38:07,458 INFO [ctc_att_transformer_decode.py:226] %WER 9.33% [4906 / 52576, 461 ins, 783 del, 3662 sub ]

# epoch 18
2021-02-17 08:38:44,550 INFO [ctc_att_transformer_decode.py:226] %WER 8.98% [4719 / 52576, 491 ins, 717 del, 3511 sub ]

# epoch 19
2021-02-17 08:39:22,562 INFO [ctc_att_transformer_decode.py:226] %WER 8.63% [4538 / 52576, 491 ins, 591 del, 3456 sub ]
```

## 2021-03-29 (Conformer)

(Fangjun): Results of <https://github.com/k2-fsa/snowfall/pull/143>

Results when adding SpecAugment with the schedule proposed in the original paper that introduces it;
best results were obtained with 80 rather than 40 filter banks.

They are obtained using the latest k2 and lhotse as of today (2021-03-29).

# average over last 5 epochs
2021-03-30 10:25:17,347 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 7.32% [3849 / 52576, 438 ins, 413 del, 2998 sub ]
2021-03-30 10:29:04,690 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 19.84% [10387 / 52343, 975 ins, 1552 del, 7860 sub ]

Epoch 0:
2021-03-30 14:05:50,210 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 23.23% [12212 / 52576, 691 ins, 2818 del, 8703 sub ]
2021-03-30 14:09:40,577 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 47.27% [24742 / 52343, 931 ins, 6718 del, 17093 sub ]

Epoch 1:
2021-03-30 10:45:30,596 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 16.31% [8574 / 52576, 637 ins, 1643 del, 6294 sub ]
2021-03-30 10:49:17,734 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 38.39% [20096 / 52343, 1092 ins, 4732 del, 14272 sub ]

Epoch 2:
2021-03-30 10:53:46,940 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 12.99% [6831 / 52576, 546 ins, 1192 del, 5093 sub ]
2021-03-30 10:57:36,322 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 33.15% [17350 / 52343, 999 ins, 3996 del, 12355 sub ]

Epoch 3:
2021-03-30 11:02:07,617 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 10.19% [5356 / 52576, 474 ins, 845 del, 4037 sub ]
2021-03-30 11:05:55,902 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 27.00% [14133 / 52343, 943 ins, 3082 del, 10108 sub ]

Epoch 4:
2021-03-30 11:10:27,557 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 9.56% [5026 / 52576, 447 ins, 792 del, 3787 sub ]
2021-03-30 11:13:47,567 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 25.45% [13319 / 52343, 795 ins, 3154 del, 9370 sub ]

Epoch 5:
2021-03-30 11:18:17,627 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 9.06% [4763 / 52576, 431 ins, 718 del, 3614 sub ]
2021-03-30 11:22:08,528 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 24.07% [12598 / 52343, 929 ins, 2538 del, 9131 sub ]

Epoch 6:
2021-03-30 11:26:39,332 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 8.63% [4535 / 52576, 463 ins, 579 del, 3493 sub ]
2021-03-30 11:30:29,829 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 23.51% [12304 / 52343, 933 ins, 2296 del, 9075 sub ]

Epoch 7:
2021-03-30 11:35:03,633 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 8.54% [4488 / 52576, 414 ins, 596 del, 3478 sub ]
2021-03-30 11:38:25,811 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 22.51% [11784 / 52343, 908 ins, 2152 del, 8724 sub ]

Epoch 8:
2021-03-30 11:42:56,193 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 7.78% [4091 / 52576, 455 ins, 481 del, 3155 sub ]
2021-03-30 11:46:46,510 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 21.36% [11183 / 52343, 920 ins, 1912 del, 8351 sub ]

Epoch 9:
2021-03-30 11:51:15,704 INFO [ctc_att_transformer_decode.py:256] [test-clean] %WER 7.88% [4142 / 52576, 406 ins, 537 del, 3199 sub ]
2021-03-30 11:55:06,389 INFO [ctc_att_transformer_decode.py:256] [test-other] %WER 21.54% [11275 / 52343, 865 ins, 2043 del, 8367 sub ]


# LibriSpeech MMI training results (Transformer)

## 2021-03-08

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/119>

TensorBoard log is available at <https://tensorboard.dev/experiment/oabLDpTJSEObc3rrWyQJig/#scalars>
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/6116390/log-train-2021-03-07-02-27-09.txt>.

Decoding results (WER) of final model averaged over last 5 epochs (i.e. epochs 5 to 9.) and each epoch model without model averaging are
listed below.

```
# average over last 5 epochs
2021-03-07 11:20:39,733 INFO [mmi_att_transformer_decode.py:312] %WER 7.72% [4057 / 52576, 578 ins, 350 del, 3129 sub ]

# epoch 0
2021-03-07 10:55:45,002 INFO [mmi_att_transformer_decode.py:312] %WER 28.36% [14909 / 52576, 1454 ins, 2461 del, 10994 sub ]

# epoch 1
2021-03-07 10:56:40,550 INFO [mmi_att_transformer_decode.py:312] %WER 14.25% [7494 / 52576, 694 ins, 1185 del, 5615 sub ]

# epoch 2
2021-03-07 10:57:33,340 INFO [mmi_att_transformer_decode.py:312] %WER 10.16% [5341 / 52576, 588 ins, 689 del, 4064 sub ]

# epoch 3
2021-03-07 10:58:22,864 INFO [mmi_att_transformer_decode.py:312] %WER 9.27% [4875 / 52576, 542 ins, 642 del, 3691 sub ]

# epoch 4
2021-03-07 10:59:11,151 INFO [mmi_att_transformer_decode.py:312] %WER 9.05% [4758 / 52576, 479 ins, 701 del, 3578 sub ]

# epoch 5
2021-03-07 10:59:59,171 INFO [mmi_att_transformer_decode.py:312] %WER 8.46% [4448 / 52576, 565 ins, 478 del, 3405 sub ]

# epoch 6
2021-03-07 11:00:46,395 INFO [mmi_att_transformer_decode.py:312] %WER 8.31% [4367 / 52576, 564 ins, 464 del, 3339 sub ]

# epoch 7
2021-03-07 11:01:32,543 INFO [mmi_att_transformer_decode.py:312] %WER 8.19% [4308 / 52576, 587 ins, 397 del, 3324 sub ]

# epoch 8
2021-03-07 11:02:19,537 INFO [mmi_att_transformer_decode.py:312] %WER 8.09% [4256 / 52576, 594 ins, 381 del, 3281 sub ]

# epoch 9
2021-03-07 11:03:04,628 INFO [mmi_att_transformer_decode.py:312] %WER 8.04% [4227 / 52576, 554 ins, 418 del, 3255 sub ]
```

# LibriSpeech MMI training results (Conformer)

## 2021-05-02

(Han Zhu): Results with VGG frontend.

Training log and tensorboard log can be found at <https://github.com/k2-fsa/snowfall/pull/182>.

Decoding results (WER) of final model averaged over last 5 epochs (i.e. epochs 5 to 9.) and each epoch model without model averaging are
listed below.

```
# average over last 5 epochs (LM rescoring with whole lattice)
2021-05-02 00:36:42,886 INFO [common.py:381] [test-clean] %WER 5.55% [2916 / 52576, 548 ins, 172 del, 2196 sub ]
2021-05-02 00:47:15,544 INFO [common.py:381] [test-other] %WER 15.32% [8021 / 52343, 1270 ins, 501 del, 6250 sub ]

# average over last 5 epochs
2021-05-01 23:35:17,891 INFO [common.py:381] [test-clean] %WER 6.65% [3494 / 52576, 457 ins, 293 del, 2744 sub ]
2021-05-01 23:37:23,141 INFO [common.py:381] [test-other] %WER 17.68% [9252 / 52343, 1020 ins, 858 del, 7374 sub ]

# epoch 0
2021-05-02 01:09:52,745 INFO [common.py:381] [test-clean] %WER 21.68% [11396 / 52576, 1438 ins, 998 del, 8960 sub ]
2021-05-02 01:11:14,618 INFO [common.py:381] [test-other] %WER 45.48% [23808 / 52343, 2571 ins, 2370 del, 18867 sub ]

# epoch 1
2021-05-02 01:12:49,179 INFO [common.py:381] [test-clean] %WER 11.76% [6184 / 52576, 695 ins, 683 del, 4806 sub ]
2021-05-02 01:14:11,675 INFO [common.py:381] [test-other] %WER 29.74% [15569 / 52343, 1442 ins, 1937 del, 12190 sub ]

# epoch 2
2021-05-02 01:15:46,336 INFO [common.py:381] [test-clean] %WER 9.45% [4966 / 52576, 552 ins, 487 del, 3927 sub ]
2021-05-02 01:17:08,992 INFO [common.py:381] [test-other] %WER 24.86% [13013 / 52343, 1194 ins, 1685 del, 10134 sub ]

# epoch 3
2021-05-02 01:18:43,584 INFO [common.py:381] [test-clean] %WER 9.49% [4987 / 52576, 549 ins, 686 del, 3752 sub ]
2021-05-02 01:20:08,417 INFO [common.py:381] [test-other] %WER 25.26% [13220 / 52343, 1029 ins, 2292 del, 9899 sub ]

# epoch 4
2021-05-02 01:21:43,498 INFO [common.py:381] [test-clean] %WER 8.00% [4207 / 52576, 492 ins, 382 del, 3333 sub ]
2021-05-02 01:23:06,132 INFO [common.py:381] [test-other] %WER 20.88% [10929 / 52343, 1056 ins, 1188 del, 8685 sub ]

# epoch 5
2021-05-02 01:24:39,382 INFO [common.py:381] [test-clean] %WER 7.89% [4148 / 52576, 500 ins, 347 del, 3301 sub ]
2021-05-02 01:26:02,202 INFO [common.py:381] [test-other] %WER 21.10% [11043 / 52343, 1233 ins, 1105 del, 8705 sub ]

# epoch 6
2021-05-02 01:27:35,616 INFO [common.py:381] [test-clean] %WER 7.72% [4058 / 52576, 471 ins, 380 del, 3207 sub ]
2021-05-02 01:28:58,678 INFO [common.py:381] [test-other] %WER 20.40% [10677 / 52343, 1106 ins, 1174 del, 8397 sub ]

# epoch 7
2021-05-02 01:30:32,897 INFO [common.py:381] [test-clean] %WER 7.40% [3893 / 52576, 470 ins, 349 del, 3074 sub ]
2021-05-02 01:31:54,306 INFO [common.py:381] [test-other] %WER 19.61% [10264 / 52343, 1037 ins, 1047 del, 8180 sub ]

# epoch 8
2021-05-02 01:33:28,578 INFO [common.py:381] [test-clean] %WER 7.40% [3890 / 52576, 489 ins, 329 del, 3072 sub ]
2021-05-02 01:34:52,473 INFO [common.py:381] [test-other] %WER 19.70% [10312 / 52343, 1157 ins, 1009 del, 8146 sub ]

# epoch 9
2021-05-02 01:36:30,299 INFO [common.py:381] [test-clean] %WER 7.32% [3848 / 52576, 525 ins, 321 del, 3002 sub ]
2021-05-02 01:37:52,445 INFO [common.py:381] [test-other] %WER 19.93% [10430 / 52343, 1251 ins, 956 del, 8223 sub ]
```

## 2021-03-26

Results when adding SpecAugment with the schedule proposed in the original paper that introduces it;
best results were obtained with 80 rather than 40 filter banks.

Training log and tensorboard log can be found at <https://github.com/k2-fsa/snowfall/pull/143>.

Average over last 5 epochs
2021-03-26 19:23:07,097 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 6.90% [3627 / 52576, 502 ins, 310 del, 2815 sub ]
2021-03-26 19:24:50,661 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 17.89% [9363 / 52343, 1141 ins, 908 del, 7314 sub ]

Epoch 1:
2021-03-26 17:28:44,723 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 21.89% [11511 / 52576, 1433 ins, 1149 del, 8929 sub ]
2021-03-26 17:31:22,335 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 45.15% [23634 / 52343, 2475 ins, 2830 del, 18329 sub ]
Epoch 2:
2021-03-26 17:33:09,657 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 12.52% [6584 / 52576, 581 ins, 899 del, 5104 sub ]
2021-03-26 17:34:32,237 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 29.97% [15689 / 52343, 1224 ins, 2419 del, 12046 sub ]
Epoch 3:
2021-03-26 17:36:27,185 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 9.96% [5234 / 52576, 680 ins, 433 del, 4121 sub ]
2021-03-26 17:37:48,597 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 25.27% [13229 / 52343, 1615 ins, 1282 del, 10332 sub ]
Epoch 4:
2021-03-26 17:39:45,618 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 8.83% [4640 / 52576, 677 ins, 333 del, 3630 sub ]
2021-03-26 17:41:08,419 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 23.00% [12038 / 52343, 1555 ins, 1070 del, 9413 sub ]
Epoch 5:
2021-03-26 17:43:01,038 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 8.65% [4548 / 52576, 595 ins, 415 del, 3538 sub ]
2021-03-26 17:44:22,546 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 21.88% [11454 / 52343, 1312 ins, 1236 del, 8906 sub ]
Epoch 6:
2021-03-26 17:46:10,045 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 7.81% [4105 / 52576, 528 ins, 354 del, 3223 sub ]
2021-03-26 17:47:30,558 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 20.72% [10844 / 52343, 1247 ins, 1133 del, 8464 sub ]
Epoch 7:
2021-03-26 17:49:34,328 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 7.94% [4174 / 52576, 534 ins, 355 del, 3285 sub ]
2021-03-26 17:50:57,266 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 20.97% [10978 / 52343, 1220 ins, 1182 del, 8576 sub ]
Epoch 8:
2021-03-26 17:53:00,601 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 7.77% [4084 / 52576, 533 ins, 336 del, 3215 sub ]
2021-03-26 17:54:21,968 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 20.42% [10687 / 52343, 1260 ins, 1059 del, 8368 sub ]
Epoch 9:
2021-03-26 18:02:31,839 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 7.52% [3954 / 52576, 497 ins, 374 del, 3083 sub ]
2021-03-26 18:04:16,252 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 20.06% [10501 / 52343, 1075 ins, 1268 del, 8158 sub ]
Epoch 10:
2021-03-26 19:42:11,096 INFO [mmi_att_transformer_decode.py:331] [test-clean] %WER 7.62% [4005 / 52576, 562 ins, 316 del, 3127 sub ]
2021-03-26 19:43:18,106 INFO [mmi_att_transformer_decode.py:331] [test-other] %WER 19.92% [10427 / 52343, 1250 ins, 1037 del, 8140 sub ]

Average over last 5 epochs, When using 40 filter banks instead of 80 (also twice smaller feature mask sizes for specaug):
2021-03-25 21:52:23,645 INFO [mmi_att_transformer_decode.py:329] [test-clean] %WER 6.93% [3645 / 52576, 529 ins, 308 del, 2808 sub ]
2021-03-25 21:53:10,674 INFO [mmi_att_transformer_decode.py:329] [test-other] %WER 18.53% [9697 / 52343, 1136 ins, 929 del, 7632 sub ]

## 2021-04-11

By Fangjun.

### Average over last 5 epochs

#### LM rescoring with whole lattice

```
$ ./mmi_att_transformer_decode.py --use-lm-rescoring=1 --num-path=-1 --max-duration=10 --output-beam-size=8

2021-04-11 10:37:58,913 INFO [common.py:356] [test-clean] %WER 5.72% [3008 / 52576, 562 ins, 164 del, 2282 sub ]
2021-04-11 10:46:03,670 INFO [common.py:356] [test-other] %WER 15.71% [8224 / 52343, 1331 ins, 562 del, 6331 sub ]
```

#### LM rescoring with n-best list

```
$ ./mmi_att_transformer_decode.py --use-lm-rescoring=1 --num-path=100 --max-duration=500 --output-beam-size=20

2021-04-11 15:17:07,792 INFO [common.py:356] [test-clean] %WER 6.31% [3316 / 52576, 746 ins, 160 del, 2410 sub ]
2021-04-11 15:19:48,583 INFO [common.py:356] [test-other] %WER 16.93% [8863 / 52343, 1649 ins, 514 del, 6700 sub ]
```



## 2021-03-08

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/119>

TensorBoard log is available at <https://tensorboard.dev/experiment/CpiZPPV6Snq3TgfDR9iJZA/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/6116417/log-train-2021-03-07-02-26-29.txt>.

Decoding results (WER) of final model averaged over last 5 epochs (i.e. epochs 5 to 9.) and each epoch model without model averaging are
listed below.

```
# average over last 5 epochs
2021-03-07 14:57:38,253 INFO [mmi_att_conformer_decode.py:312] %WER 6.84% [3598 / 52576, 538 ins, 257 del, 2803 sub ]

# epoch 0
2021-03-07 14:55:05,898 INFO [mmi_att_conformer_decode.py:312] %WER 20.76% [10914 / 52576, 1543 ins, 921 del, 8450 sub ]

# epoch 1
2021-03-07 14:56:18,627 INFO [mmi_att_conformer_decode.py:312] %WER 12.19% [6409 / 52576, 477 ins, 1108 del, 4824 sub ]

# epoch 2
2021-03-07 14:57:26,384 INFO [mmi_att_conformer_decode.py:312] %WER 9.50% [4997 / 52576, 596 ins, 447 del, 3954 sub ]

# epoch 3
2021-03-07 14:58:31,390 INFO [mmi_att_conformer_decode.py:312] %WER 8.96% [4711 / 52576, 588 ins, 396 del, 3727 sub ]

# epoch 4
2021-03-07 14:59:36,554 INFO [mmi_att_conformer_decode.py:312] %WER 8.36% [4393 / 52576, 564 ins, 372 del, 3457 sub ]

# epoch 5
2021-03-07 15:00:40,190 INFO [mmi_att_conformer_decode.py:312] %WER 8.06% [4239 / 52576, 613 ins, 348 del, 3278 sub ]

# epoch 6
2021-03-07 15:01:42,861 INFO [mmi_att_conformer_decode.py:312] %WER 7.95% [4181 / 52576, 599 ins, 326 del, 3256 sub ]

# epoch 7
2021-03-07 15:02:43,487 INFO [mmi_att_conformer_decode.py:312] %WER 8.00% [4207 / 52576, 644 ins, 306 del, 3257 sub ]

# epoch 8
2021-03-07 15:03:45,646 INFO [mmi_att_conformer_decode.py:312] %WER 7.74% [4072 / 52576, 536 ins, 341 del, 3195 sub ]

# epoch 9
2021-03-07 15:04:47,523 INFO [mmi_att_conformer_decode.py:312] %WER 7.79% [4095 / 52576, 677 ins, 280 del, 3138 sub ]
```


# LibriSpeech MMI training results (ContextNet)

## 2021-04-29

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/173>

TensorBoard log is available at <https://tensorboard.dev/experiment/Wka3gjonTzKa1cL7gXPpag/>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/6395833/log-train-0-2021-04-27-01-30-48.txt>.

Results could be reproduced with: 
```
python mmi_att_transformer_train.py --model-type contextnet --lr-factor 2.0 --weight-decay 1e-6 --max-duration 300 --full-libri 0 --use-ali-model 0
```

Decoding results (WER) of final model averaged over last 5 epochs (i.e. epochs 5 to 9.)
and each epoch model without model averaging are listed below.

```
# average over last 5 epochs (LM rescoring with whole lattice)
2021-05-06 02:44:10,156 INFO [common.py:381] [test-clean] %WER 6.66% [3504 / 52576, 767 ins, 190 del, 2547 sub ]
2021-05-06 03:05:42,063 INFO [common.py:381] [test-other] %WER 18.36% [9612 / 52343, 1651 ins, 683 del, 7278 sub ]

# average over last 5 epochs
2021-04-27 12:48:07,217 INFO [common.py:365] [test-clean] %WER 8.03% [4220 / 52576, 570 ins, 366 del, 3284 sub ]
2021-04-27 12:49:27,507 INFO [common.py:365] [test-other] %WER 22.14% [11588 / 52343, 1232 ins, 1350 del, 9006 sub ]

# epoch 0
2021-04-29 01:46:31,265 INFO [common.py:365] [test-clean] %WER 13.48% [7088 / 52576, 767 ins, 744 del, 5577 sub ]
2021-04-29 01:47:50,590 INFO [common.py:365] [test-other] %WER 35.13% [18390 / 52343, 1596 ins, 2473 del, 14321 sub ]

# epoch 1
2021-04-29 01:49:23,170 INFO [common.py:365] [test-clean] %WER 10.28% [5405 / 52576, 568 ins, 601 del, 4236 sub ]
2021-04-29 01:50:44,332 INFO [common.py:365] [test-other] %WER 28.70% [15025 / 52343, 1223 ins, 2089 del, 11713 sub ]

# epoch 2
2021-04-29 01:52:14,942 INFO [common.py:365] [test-clean] %WER 9.74% [5121 / 52576, 587 ins, 536 del, 3998 sub ]
2021-04-29 01:53:34,017 INFO [common.py:365] [test-other] %WER 26.25% [13740 / 52343, 1258 ins, 1732 del, 10750 sub ]

# epoch 3
2021-04-29 01:55:05,413 INFO [common.py:365] [test-clean] %WER 9.19% [4830 / 52576, 598 ins, 447 del, 3785 sub ]
2021-04-29 01:56:25,599 INFO [common.py:365] [test-other] %WER 25.09% [13134 / 52343, 1372 ins, 1533 del, 10229 sub ]

# epoch 4
2021-04-29 01:57:55,792 INFO [common.py:365] [test-clean] %WER 9.25% [4863 / 52576, 561 ins, 500 del, 3802 sub ]
2021-04-29 01:59:15,584 INFO [common.py:365] [test-other] %WER 25.37% [13281 / 52343, 1194 ins, 1927 del, 10160 sub ]

# epoch 5
2021-04-29 02:00:48,407 INFO [common.py:365] [test-clean] %WER 8.76% [4606 / 52576, 573 ins, 423 del, 3610 sub ]
2021-04-29 02:02:09,057 INFO [common.py:365] [test-other] %WER 24.35% [12744 / 52343, 1250 ins, 1649 del, 9845 sub ]

# epoch 6
2021-04-29 02:03:41,794 INFO [common.py:365] [test-clean] %WER 8.87% [4666 / 52576, 584 ins, 459 del, 3623 sub ]
2021-04-29 02:05:00,733 INFO [common.py:365] [test-other] %WER 24.82% [12994 / 52343, 1258 ins, 1779 del, 9957 sub ]

# epoch 7
2021-04-29 02:06:32,426 INFO [common.py:365] [test-clean] %WER 8.85% [4655 / 52576, 629 ins, 419 del, 3607 sub ]
2021-04-29 02:07:53,611 INFO [common.py:365] [test-other] %WER 24.78% [12971 / 52343, 1389 ins, 1504 del, 10078 sub ]

# epoch 8
2021-04-29 02:09:25,071 INFO [common.py:365] [test-clean] %WER 8.70% [4572 / 52576, 628 ins, 403 del, 3541 sub ]
2021-04-29 02:10:44,437 INFO [common.py:365] [test-other] %WER 23.76% [12435 / 52343, 1350 ins, 1360 del, 9725 sub ]

# epoch 9
2021-04-29 02:12:15,899 INFO [common.py:365] [test-clean] %WER 8.70% [4572 / 52576, 702 ins, 395 del, 3475 sub ]
2021-04-29 02:13:36,382 INFO [common.py:365] [test-other] %WER 24.07% [12597 / 52343, 1482 ins, 1424 del, 9691 sub ]
```
