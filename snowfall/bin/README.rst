Introduction
============

This document describes how to install and use the utility ``snowfall``.

Installation
------------

You can use **one** of the following three ways to install ``snowfall``:

- (1) Run:

  .. code-block:: bash

    cd /path/to/snowfall
    python3 setup.py install

- (2) Use:

  .. code-block:: bash

    $ cd /path/to/snowfall
    $ pip install -e .

- (3) Modify the environment variable ``PYTHONPATH`` and ``PATH``:

  .. code-block:: bash

    $ export PYTHONPATH=/path/to/snowfall:$PYTHONPATH
    $ export PYTHONPATH=/path/to/lhose:$PYTHONPATH
    $ export PYTHONPATH=/path/to/k2/k2/python:$PYTHONPATH
    $ export PYTHONPATH=/path/to/k2/k2/build_release/lib:$PYTHONPATH

    $ export PATH=/path/to/snowfall/snowfall/bin:$PATH


To check that you have installed it successfully, please run:

  .. code-block:: bash

    $ which snowfall

It should give you the path to ``snowfall``. For instance, the output on my computer is:

  .. code-block:: bash

    /Users/fangjun/open-source/snowfall/snowfall/bin/snowfall

Usage
-----

.. code-block:: bash

  $ snowfall --help

It prints something like below:

  .. code-block:: bash

    Usage: snowfall [OPTIONS] COMMAND [ARGS]...

      Entry point to the collection of utilities in snowall.

    Options:
      --help  Show this message and exit.

    Commands:
      ali  Alignment tools in snowfall

We will extend the available commands along the way. To run the command ``ali``, use:

  .. code-block:: bash

    $ snowfall ali --help

It prints:

  .. code-block:: bash

    Usage: snowfall ali [OPTIONS] COMMAND [ARGS]...

      Alignment tools in snowfall

    Options:
      --help  Show this message and exit.

    Commands:
      edit-distance  Compute edit distance between two alignments.

To run the ``edit-distance`` command inside ``ali``, use:

  .. code-block:: bash

    $ snowfall ali edit-distance --help

Its output is:

  .. code-block:: bash

    Usage: snowfall ali edit-distance [OPTIONS]

      Compute edit distance between two alignments.

      The reference/hypothesis alignment file contains a python object Dict[str,
      Alignment] and it can be loaded using `torch.load`. The dict is indexed by
      utterance ID.

      The symbol table, if provided, has the following format for each line:

          symbol integer_id

      It can be loaded by `k2.SymbolTable.from_file()`.

    Options:
      -r, --ref FILE           The file containing reference alignments
                               [required]
      -h, --hyp FILE           The file containing hypothesis alignments
                               [required]
      -t, --type TEXT          The type of the alignment to use for computing the
                               edit distance  [required]
      -o, --output-file FILE   Output file  [required]
      -s, --symbol-table FILE  The symbol table for the given type of alignment
      --help                   Show this message and exit.

We provide some test data files for the ``edit-distance`` command. To test it, run:

  .. code-block::

    $ cd /path/to/snowfall/test/fixtures
    $ ./gen_ali.py
    $ snowfall ali edit-distance -r ./ref.pt -h ./hyp.pt -t phone -s ./sym.txt -o out.txt

You will find the output in ``out.txt``.
