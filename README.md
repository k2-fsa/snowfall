# k2_librispeech
An example of how to build G and L FST for K2.

Most scripts of this example are copied from Kaldi.

## Results
```bash
$ ./run.sh
```

```plaintext
Downloading file '3-gram.pruned.1e-7.arpa.gz' into 'data/local/lm'...
'3-gram.pruned.1e-7.arpa.gz' already exists and appears to be complete
Downloading file 'librispeech-vocab.txt' into 'data/local/lm'...
'librispeech-vocab.txt' already exists and appears to be complete
Downloading file 'librispeech-lexicon.txt' into 'data/local/lm'...
'librispeech-lexicon.txt' already exists and appears to be complete
Preparing phone lists
Lexicon text file saved as: data/local/dict_nosp/lexicon.txt
local/prepare_lang.sh data/local/dict_nosp <UNK> data/local/lang_tmp_nosp data/lang_nosp
```

```bash
$ tree
```

```plaintext
.
├── README.md
├── data
│   ├── lang_nosp
│   │   ├── L.fst.txt
│   │   ├── L_disambig.fst.txt
│   │   ├── oov.int
│   │   ├── oov.txt
│   │   ├── phones
│   │   │   ├── disambig.csl
│   │   │   ├── disambig.int
│   │   │   ├── disambig.txt
│   │   │   ├── extra_questions.txt
│   │   │   ├── nonsilence.csl
│   │   │   ├── nonsilence.int
│   │   │   ├── nonsilence.txt
│   │   │   ├── optional_silence.csl
│   │   │   ├── optional_silence.int
│   │   │   ├── optional_silence.txt
│   │   │   ├── silence.csl
│   │   │   ├── silence.int
│   │   │   ├── silence.txt
│   │   │   ├── wdisambig.txt
│   │   │   ├── wdisambig_phones.int
│   │   │   ├── wdisambig_words.int
│   │   │   ├── word_boundary.int
│   │   │   └── word_boundary.txt
│   │   ├── phones.txt
│   │   └── words.txt
│   └── local
│       ├── dict_nosp
│       │   ├── lexicon.txt
│       │   ├── lexicon_raw_nosil.txt
│       │   ├── lexiconp.txt
│       │   ├── nonsilence_phones.txt
│       │   ├── optional_silence.txt
│       │   └── silence_phones.txt
│       ├── lang_tmp_nosp
│       │   ├── lex_ndisambig
│       │   ├── lexiconp.txt
│       │   ├── lexiconp_disambig.txt
│       │   └── phone_map.txt
│       └── lm
│           ├── 3-gram.pruned.1e-7.arpa.gz
│           ├── librispeech-lexicon.txt
│           ├── librispeech-vocab.txt
│           └── lm_tgmed.arpa.gz -> 3-gram.pruned.1e-7.arpa.gz
├── local
│   ├── add_lex_disambig.pl
│   ├── apply_map.pl
│   ├── download_lm.sh
│   ├── make_lexicon_fst.py
│   ├── parse_options.sh
│   ├── prepare_dict.sh
│   ├── prepare_lang.sh
│   └── sym2int.pl
└── run.sh

8 directories, 48 files
```
