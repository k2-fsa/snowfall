## MMI+MBR training
First try of MMI+MBR training. The following result using
`epoch-9.pt` is obtained by Fangjun.

```
2021-02-01 08:09:31,706 INFO [mmi_mbr_decode.py:295] %WER 11.04% [5805 / 52576, 802 ins, 535 del, 4468 sub ]
```

# LibriSpeech CTC training results (TDNN-LSTM)

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/l3dVVLgBRIOqBY4u3e4dbA/#scalars>
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/5995756/log-train-2021-02-16-12-36-09.txt>.

Decoding results (WER) of each epoch are listed below.
They are obtained using the latest k2 and lhotse as of today (2021-02-17).
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
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/5995759/log-train-2021-02-16-12-36-23.txt>.

Decoding results (WER) of each epoch are listed below.
They are obtained using the latest k2 and lhotse as of today (2021-02-17).

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

# LibriSpeech MMI+MBR training results (TDNN-LSTM)

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/fYDCXr2iQWOwZQYQXTOZLQ/#scalars>
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/5995761/log-train-2021-02-17-01-10-23.txt>.

Decoding results (WER) of each epoch are listed below.
They are obtained using the latest k2 and lhotse as of today (2021-02-17).

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
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/5995766/log-train-2021-02-16-02-27-50.txt>.

Decoding results (WER) of final model averaged over last 5 epochs and each epoch model without model averaging are listed below.
They are obtained using the latest k2 and lhotse as of today (2021-02-17).

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

## 2021-02-17

(Han Zhu): Results of <https://github.com/k2-fsa/snowfall/pull/103>

TensorBoard log is available at <https://tensorboard.dev/experiment/KnOfJepMTKyYcy1c3ziMgQ/#scalars>
and the training log can be downloaded using <https://github.com/k2-fsa/snowfall/files/5995771/log-train-2021-02-17-01-45-18.txt>.

Decoding results (WER) of final model averaged over last 5 epochs and each epoch model without model averaging are listed below.
They are obtained using the latest k2 and lhotse as of today (2021-02-17).

```
# average over last 5 epochs
2021-02-17 09:38:03,597 INFO [mmi_att_transformer_decode.py:300] %WER 8.05% [4230 / 52576, 610 ins, 367 del, 3253 sub ]

# epoch 0
2021-02-17 09:19:40,795 INFO [mmi_att_transformer_decode.py:300] %WER 66.80% [35122 / 52576, 2335 ins, 8193 del, 24594 sub ]

# epoch 1
2021-02-17 09:20:39,043 INFO [mmi_att_transformer_decode.py:300] %WER 34.26% [18010 / 52576, 1964 ins, 2690 del, 13356 sub ]

# epoch 2
2021-02-17 09:21:35,649 INFO [mmi_att_transformer_decode.py:300] %WER 24.32% [12786 / 52576, 1482 ins, 1734 del, 9570 sub ]

# epoch 3
2021-02-17 09:22:31,828 INFO [mmi_att_transformer_decode.py:300] %WER 19.61% [10309 / 52576, 1270 ins, 1330 del, 7709 sub ]

# epoch 4
2021-02-17 09:23:26,471 INFO [mmi_att_transformer_decode.py:300] %WER 16.28% [8560 / 52576, 955 ins, 1191 del, 6414 sub ]

# epoch 5
2021-02-17 09:24:19,785 INFO [mmi_att_transformer_decode.py:300] %WER 12.54% [6595 / 52576, 723 ins, 814 del, 5058 sub ]

# epoch 6
2021-02-17 09:25:12,438 INFO [mmi_att_transformer_decode.py:300] %WER 10.46% [5500 / 52576, 653 ins, 618 del, 4229 sub ]

# epoch 7
2021-02-17 09:26:04,223 INFO [mmi_att_transformer_decode.py:300] %WER 9.61% [5054 / 52576, 604 ins, 531 del, 3919 sub ]

# epoch 8
2021-02-17 09:26:55,382 INFO [mmi_att_transformer_decode.py:300] %WER 9.26% [4868 / 52576, 553 ins, 539 del, 3776 sub ]

# epoch 9
2021-02-17 09:27:46,199 INFO [mmi_att_transformer_decode.py:300] %WER 9.05% [4757 / 52576, 577 ins, 503 del, 3677 sub ]

# epoch 10
2021-02-17 09:28:35,118 INFO [mmi_att_transformer_decode.py:300] %WER 8.95% [4703 / 52576, 607 ins, 475 del, 3621 sub ]

# epoch 11
2021-02-17 09:29:22,655 INFO [mmi_att_transformer_decode.py:300] %WER 8.60% [4521 / 52576, 619 ins, 384 del, 3518 sub ]

# epoch 12
2021-02-17 09:30:10,919 INFO [mmi_att_transformer_decode.py:300] %WER 8.52% [4477 / 52576, 588 ins, 411 del, 3478 sub ]

# epoch 13
2021-02-17 09:30:57,622 INFO [mmi_att_transformer_decode.py:300] %WER 8.49% [4463 / 52576, 555 ins, 441 del, 3467 sub ]

# epoch 14
2021-02-17 09:31:43,641 INFO [mmi_att_transformer_decode.py:300] %WER 8.17% [4295 / 52576, 581 ins, 381 del, 3333 sub ]

# epoch 15
2021-02-17 09:32:30,313 INFO [mmi_att_transformer_decode.py:300] %WER 8.33% [4378 / 52576, 613 ins, 406 del, 3359 sub ]

# epoch 16
2021-02-17 09:33:17,155 INFO [mmi_att_transformer_decode.py:300] %WER 8.49% [4465 / 52576, 610 ins, 385 del, 3470 sub ]

# epoch 17
2021-02-17 09:34:04,108 INFO [mmi_att_transformer_decode.py:300] %WER 8.27% [4346 / 52576, 617 ins, 361 del, 3368 sub ]

# epoch 18
2021-02-17 09:34:49,242 INFO [mmi_att_transformer_decode.py:300] %WER 8.33% [4382 / 52576, 629 ins, 361 del, 3392 sub ]

# epoch 19
2021-02-17 09:35:33,273 INFO [mmi_att_transformer_decode.py:300] %WER 8.39% [4409 / 52576, 638 ins, 378 del, 3393 sub ]
```