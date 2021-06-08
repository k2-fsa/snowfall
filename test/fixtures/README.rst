Introduction
============

gen_ali.py
----------

This file generates test data for ``snowfall ali edit-distance``.
Use it in the following way:

  .. code-block:: bash

    $ ./gen_ali.py
    $ snowfall ali edit-distance -r ./ref.pt -h ./hyp.pt -t phone -s ./sym.txt -o a.txt

a.flac, a.TextGrid
------------------
These two files are test data for `snowfall ali visualize`.
Use it in the following way:

  .. code-block:: bash

    $ snowfall ali visualize -i ./a.flac -t a.TextGrid -w 20 -s 0.5 -e 3.5 -o a-0.5-3.5.pdf
    $ snowfall ali visualize -i ./a.flac -t a.TextGrid -w 100 -o all.pdf
