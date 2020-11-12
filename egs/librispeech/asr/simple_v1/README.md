

## k2_librispeech
An example of how to build G and L FST for K2.

Most scripts of this example are copied from Kaldi.

### Run scripts
```bash
$ ./run.sh

$ ls data/lang_nosp

G.fsa.txt
L.fst.txt
L_disambig.fst.txt
oov.int
oov.txt
phones
phones.txt
words.txt
```

### Load L, G into K2
```python
import k2, _k2


with open('data/lang_nosp/L.fst.txt') as f:
    s = f.read()

Lfst = k2.Fsa.from_openfst(s, acceptor=False)

with open('data/lang_nosp/G.fsa.txt') as f:
    s = f.read()

Gfsa = k2.Fsa.from_openfst(s, acceptor=True)
```

### An example of G building
The `toy.arpa` file:
```plain
\data\
ngram 1=5
ngram 2=6
ngram 3=1

\1-grams:
-2.348754 </s>
-99 <s> -1.070027
-4.214113 A -0.5964623
-4.255245 B -0.3214741
-4.20255  C -0.2937318

\2-grams:
-4.284099 <s> A -0.1969815
-1.100091 A </s>
-2.839235 A B -0.1747991
-2.838903 A C -0.5100551
-1.104238 B </s>
-1.251644 C </s>

\3-grams:
-0.1605104  A C B

\end\
```

Build G fst:
```bash
$ local/arpa2fst.py toy.arpa

0 1 </s> -5.408205947510138
2 0 <eps> -1.070027
0 3 A -9.703353773992418
3 0 <eps> -0.5964623
0 4 B -9.79806370403745
4 0 <eps> -0.3214741
0 5 C -9.676728982562127
5 0 <eps> -0.2937318
2 6 A -9.864502494310699
6 3 <eps> -0.1969815
3 1 </s> -2.5330531375369127
3 7 B -6.53758018650695
7 4 <eps> -0.1747991
3 8 C -6.536815728256077
8 5 <eps> -0.5100551
4 1 </s> -2.5426019579175594
5 1 </s> -2.8820168161354394
8 9 B -0.36958885431051147
1
```

Draw it by Graphviz:
```
digraph FST {
rankdir = LR;
size = "8.5,11";
label = "";
center = 1;
ranksep = "0.4";
nodesep = "0.25";
0 [label = "0", shape = circle, style = bold, fontsize = 14]
	0 -> 1 [label = "</s>/-5.4082", fontsize = 14];
	0 -> 3 [label = "A/-9.7034", fontsize = 14];
	0 -> 4 [label = "B/-9.7981", fontsize = 14];
	0 -> 5 [label = "C/-9.6767", fontsize = 14];
1 [label = "1", shape = doublecircle, style = solid, fontsize = 14]
2 [label = "2", shape = circle, style = solid, fontsize = 14]
	2 -> 0 [label = "<eps>/-1.07", fontsize = 14];
	2 -> 6 [label = "A/-9.8645", fontsize = 14];
3 [label = "3", shape = circle, style = solid, fontsize = 14]
	3 -> 0 [label = "<eps>/-0.59646", fontsize = 14];
	3 -> 1 [label = "</s>/-2.5331", fontsize = 14];
	3 -> 7 [label = "B/-6.5376", fontsize = 14];
	3 -> 8 [label = "C/-6.5368", fontsize = 14];
4 [label = "4", shape = circle, style = solid, fontsize = 14]
	4 -> 0 [label = "<eps>/-0.32147", fontsize = 14];
	4 -> 1 [label = "</s>/-2.5426", fontsize = 14];
5 [label = "5", shape = circle, style = solid, fontsize = 14]
	5 -> 0 [label = "<eps>/-0.29373", fontsize = 14];
	5 -> 1 [label = "</s>/-2.882", fontsize = 14];
6 [label = "6", shape = circle, style = solid, fontsize = 14]
	6 -> 3 [label = "<eps>/-0.19698", fontsize = 14];
7 [label = "7", shape = circle, style = solid, fontsize = 14]
	7 -> 4 [label = "<eps>/-0.1748", fontsize = 14];
8 [label = "8", shape = circle, style = solid, fontsize = 14]
	8 -> 5 [label = "<eps>/-0.51006", fontsize = 14];
	8 -> 9 [label = "B/-0.36959", fontsize = 14];
9 [label = "9", shape = circle, style = solid, fontsize = 14]
}
```
