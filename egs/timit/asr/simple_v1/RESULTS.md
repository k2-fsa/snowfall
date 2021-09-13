# TIMIT CTC Training Results

## 2021-09-03
(Mingshuang Luo):

### TIMIT CTC_Train Based on 48 phones

Testing results based on different training epochs:
```
epoch=20
2021-09-03 10:54:10,903 INFO [ctc_decode.py:188] %PER 30.34% [2225 / 7333, 293 ins, 441 del, 1491 sub ]

epoch=30
2021-09-03 10:59:10,147 INFO [ctc_decode.py:188] %PER 29.77% [2183 / 7333, 221 ins, 473 del, 1489 sub ]

epoch=35
2021-09-03 11:11:00,885 INFO [ctc_decode.py:188] %PER 28.94% [2122 / 7333, 266 ins, 397 del, 1459 sub ]

epoch=40
2021-09-03 11:12:39,029 INFO [ctc_decode.py:188] %PER 29.52% [2165 / 7333, 304 ins, 348 del, 1513 sub ]
```

### TIMIT CTC_Train Based on 39 phones

Testing results based on different training epochs:
```
epoch=40
2021-09-13 11:02:14,793 INFO [ctc_decode.py:189] %PER 25.61% [1848 / 7215, 301 ins, 396 del, 1151 sub ]

epoch=45
2021-09-13 11:01:20,787 INFO [ctc_decode.py:189] %PER 25.50% [1840 / 7215, 286 ins, 386 del, 1168 sub ]

epoch=47
2021-09-13 11:04:05,533 INFO [ctc_decode.py:189] %PER 26.20% [1890 / 7215, 373 ins, 367 del, 1150 sub ]

```
### TIMIT CTC_TRAIN_with_CRDNN Based on 48 phones

Testing results based on different training epochs:
```
epoch=35
2021-09-13 11:21:01,592 INFO [ctc_crdnn_decode.py:201] %PER 20.46% [1476 / 7215, 249 ins, 356 del, 871 sub ]

epoch=45
2021-09-13 11:22:02,221 INFO [ctc_crdnn_decode.py:201] %PER 19.75% [1425 / 7215, 239 ins, 324 del, 862 sub ]

epoch=53
2021-09-13 11:23:07,969 INFO [ctc_crdnn_decode.py:201] %PER 18.86% [1361 / 7215, 214 ins, 320 del, 827 sub ]

```

### TIMIT CTC_TRAIN_with_CRDNN Based on 39 phones

Testing results based on different training epochs:
```
epoch=26
2021-09-13 11:32:41,388 INFO [ctc_crdnn_decode.py:201] %PER 21.04% [1518 / 7215, 345 ins, 251 del, 922 sub ]

epoch=45
2021-09-13 11:34:27,566 INFO [ctc_crdnn_decode.py:201] %PER 18.74% [1352 / 7215, 316 ins, 239 del, 797 sub ]

epoch=55
2021-09-13 11:35:55,751 INFO [ctc_crdnn_decode.py:201] %PER 18.24% [1316 / 7215, 267 ins, 242 del, 807 sub ]

```