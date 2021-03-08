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

# LibriSpeech MMI training results (TDNN-LSTM)

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/hPLbUWUwT06fljbvKGnlUA/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/5995759/log-train-2021-02-16-12-36-23.txt>.

Decoding results (WER) of each epoch are listed below. They are obtained using the latest k2 and lhotse as of today (
2021-02-17).

```
# epoch 0
2021-02-17 09:42:01,523 INFO [mmi_bigram_decode.py:252] %WER 16.58% [8715 / 52576, 1074 ins, 976 del, 6665 sub ]

# epoch 1
2021-02-17 09:42:52,408 INFO [mmi_bigram_decode.py:252] %WER 12.53% [6590 / 52576, 726 ins, 827 del, 5037 sub ]

# epoch 2
2021-02-17 09:43:41,617 INFO [mmi_bigram_decode.py:252] %WER 12.33% [6480 / 52576, 724 ins, 851 del, 4905 sub ]

# epoch 3
2021-02-17 09:44:30,664 INFO [mmi_bigram_decode.py:252] %WER 11.67% [6135 / 52576, 644 ins, 901 del, 4590 sub ]

# epoch 4
2021-02-17 09:45:18,406 INFO [mmi_bigram_decode.py:252] %WER 11.59% [6096 / 52576, 597 ins, 1000 del, 4499 sub ]

# epoch 5
2021-02-17 09:46:06,066 INFO [mmi_bigram_decode.py:252] %WER 10.95% [5759 / 52576, 660 ins, 641 del, 4458 sub ]

# epoch 6
2021-02-17 09:46:52,257 INFO [mmi_bigram_decode.py:252] %WER 10.54% [5542 / 52576, 638 ins, 628 del, 4276 sub ]

# epoch 7
2021-02-17 09:47:38,464 INFO [mmi_bigram_decode.py:252] %WER 10.53% [5535 / 52576, 696 ins, 585 del, 4254 sub ]

# epoch 8
2021-02-17 09:48:23,188 INFO [mmi_bigram_decode.py:252] %WER 10.27% [5401 / 52576, 716 ins, 527 del, 4158 sub ]

# epoch 9
2021-02-17 09:49:08,283 INFO [mmi_bigram_decode.py:252] %WER 10.38% [5460 / 52576, 745 ins, 496 del, 4219 sub ]
```

## 2021-02-18 (with BucketingSampler)

On GTX2080Ti with max_frames=30000 (could go up to as much as 45000 I think)

```
Epoch 9:
%WER 10.50% [5523 / 52576, 775 ins, 487 del, 4261 sub ]
```

On Tesla V100 with max_frames=130000 (max GPU memory usage), one epoch takes ~13min.

```
Epoch 0:
2021-02-18 16:08:14,155 INFO [mmi_bigram_decode.py:259] %WER 18.75% [9858 / 52576, 1324 ins, 1207 del, 7327 sub ]
Epoch 1:
2021-02-18 16:09:24,417 INFO [mmi_bigram_decode.py:259] %WER 13.77% [7238 / 52576, 962 ins, 765 del, 5511 sub ]
Epoch 2:
2021-02-18 16:10:22,321 INFO [mmi_bigram_decode.py:259] %WER 12.78% [6717 / 52576, 816 ins, 786 del, 5115 sub ]
Epoch 3:
2021-02-18 16:11:20,510 INFO [mmi_bigram_decode.py:259] %WER 12.53% [6587 / 52576, 819 ins, 745 del, 5023 sub ]
Epoch 4:
2021-02-18 16:12:35,212 INFO [mmi_bigram_decode.py:259] %WER 11.77% [6190 / 52576, 770 ins, 748 del, 4672 sub ]
Epoch 5:
2021-02-18 16:13:35,180 INFO [mmi_bigram_decode.py:259] %WER 11.14% [5857 / 52576, 830 ins, 564 del, 4463 sub ]
Epoch 6:
2021-02-18 16:14:34,464 INFO [mmi_bigram_decode.py:259] %WER 11.14% [5859 / 52576, 838 ins, 506 del, 4515 sub ]
Epoch 7:
2021-02-18 16:15:34,764 INFO [mmi_bigram_decode.py:259] %WER 10.80% [5678 / 52576, 790 ins, 569 del, 4319 sub ]
Epoch 8:
2021-02-18 16:16:32,359 INFO [mmi_bigram_decode.py:259] %WER 10.81% [5683 / 52576, 800 ins, 511 del, 4372 sub ]
Epoch 9:
2021-02-18 16:17:57,772 INFO [mmi_bigram_decode.py:259] %WER 10.62% [5584 / 52576, 819 ins, 488 del, 4277 sub ]
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

# LibriSpeech CTC training results (Transformer)

## 2021-02-17

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

# LibriSpeech MMI training results (Transformer)

## 2021-03-08

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/119>

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

## 2021-03-08

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/119>

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