# AIShell MMI training results
(Pingfeng Luo): This CER is obtained using the latest k2 and lhotse as of today (2021-02-10).

## 2021-02-10
```
# epoch 7
2021-02-10 18:12:22,691 INFO [mmi_bigram_decode.py:263] %WER 10.06% [10542 / 104765, 436 ins, 495 del, 9611 sub ] exp-lstm-adam-mmi-bigram-musan
```

# AIShell CTC training results

## 2021-02-12

(Fangjun): Results of <https://github.com/k2-fsa/snowfall/pull/99>

TensorBoard log is available at <https://tensorboard.dev/experiment/5bMFoRjVT7OMRWVFd3qVAA/#scalars>
and the training log can be downloaded
using <https://github.com/k2-fsa/snowfall/files/5971503/log-train-2021-02-12-14-19-11.txt>.

Decoding results of each epoch (the first line is WER and the second CER) are listed below. They are obtained using the
latest k2 and lhotse as of today (2021-02-12).

```
# epoch 0
2021-02-12 17:15:21,695 INFO [ctc_decode.py:182] %WER 28.79% [18552 / 64428, 1612 ins, 2177 del, 14763 sub ]
2021-02-12 17:15:21,696 INFO [ctc_decode.py:186] %WER 19.35% [20274 / 104767, 591 ins, 599 del, 19084 sub ]

# epoch 1
2021-02-12 17:18:12,417 INFO [ctc_decode.py:182] %WER 26.12% [16829 / 64428, 1454 ins, 2083 del, 13292 sub ]
2021-02-12 17:18:12,417 INFO [ctc_decode.py:186] %WER 16.75% [17548 / 104767, 385 ins, 598 del, 16565 sub ]

# epoch 2
2021-02-12 17:24:23,362 INFO [ctc_decode.py:182] %WER 24.88% [16032 / 64428, 1279 ins, 2129 del, 12624 sub ]
2021-02-12 17:24:23,362 INFO [ctc_decode.py:186] %WER 15.74% [16491 / 104767, 342 ins, 669 del, 15480 sub ]

# epoch 3
2021-02-12 17:28:05,635 INFO [ctc_decode.py:182] %WER 24.98% [16097 / 64428, 1356 ins, 2071 del, 12670 sub ]
2021-02-12 17:28:05,636 INFO [ctc_decode.py:186] %WER 15.64% [16387 / 104767, 354 ins, 602 del, 15431 sub ]

# epoch 4
2021-02-12 17:32:33,326 INFO [ctc_decode.py:182] %WER 24.57% [15827 / 64428, 1279 ins, 2009 del, 12539 sub ]
2021-02-12 17:32:33,326 INFO [ctc_decode.py:186] %WER 15.50% [16236 / 104767, 368 ins, 607 del, 15261 sub ]

# epoch 5
2021-02-12 17:34:13,417 INFO [ctc_decode.py:182] %WER 23.70% [15270 / 64428, 1241 ins, 2003 del, 12026 sub ]
2021-02-12 17:34:13,418 INFO [ctc_decode.py:186] %WER 14.63% [15332 / 104767, 356 ins, 529 del, 14447 sub ]

# epoch 6
2021-02-12 17:36:42,475 INFO [ctc_decode.py:182] %WER 24.61% [15858 / 64428, 1343 ins, 2022 del, 12493 sub ]
2021-02-12 17:36:42,475 INFO [ctc_decode.py:186] %WER 15.38% [16118 / 104767, 356 ins, 569 del, 15193 sub ]

# epoch 7
2021-02-12 17:52:25,302 INFO [ctc_decode.py:182] %WER 23.88% [15387 / 64428, 1245 ins, 1992 del, 12150 sub ]
2021-02-12 17:52:25,302 INFO [ctc_decode.py:186] %WER 14.74% [15440 / 104767, 345 ins, 548 del, 14547 sub ]

# epoch 8
2021-02-12 19:27:17,389 INFO [ctc_decode.py:182] %WER 23.89% [15389 / 64428, 1286 ins, 1967 del, 12136 sub ]
2021-02-12 19:27:17,389 INFO [ctc_decode.py:186] %WER 14.77% [15479 / 104767, 384 ins, 471 del, 14624 sub ]

# epoch 9
2021-02-12 19:28:44,843 INFO [ctc_decode.py:182] %WER 24.16% [15563 / 64428, 1271 ins, 2016 del, 12276 sub ]
2021-02-12 19:28:44,843 INFO [ctc_decode.py:186] %WER 14.95% [15662 / 104767, 403 ins, 508 del, 14751 sub ]
```
