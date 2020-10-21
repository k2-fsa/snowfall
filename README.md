# k2_librispeech
An example of how to build G and L FST for K2.

Most scripts of this example are copied from Kaldi.

## Run scripts
```bash
$ ./run.sh

Downloading file '3-gram.pruned.1e-7.arpa.gz' into 'data/local/lm'...
Downloading file 'librispeech-vocab.txt' into 'data/local/lm'...
Downloading file 'librispeech-lexicon.txt' into 'data/local/lm'...
Preparing phone lists
Lexicon text file saved as: data/local/dict_nosp/lexicon.txt
local/prepare_lang.sh data/local/dict_nosp <UNK> data/local/lang_tmp_nosp data/lang_nosp
```

```bash
$ ls data/lang_nosp

L.fst.txt               oov.int                 phones                  words.txt
L_disambig.fst.txt      oov.txt                 phones.txt
```

## Load L fst into K2
```python
import k2, _k2


with open('data/lang_nosp/L.fst.txt') as f:
    s = f.read()

fsa = k2.Fsa.from_openfst(s, acceptor=False)
```
