# Overview

This repository contains all the data and code necessary to reproduce Study 1 and Study 3 in the paper.

The abstracts for the papers are ``cscw50_papers-abstracts.csv``

The "gold-standard" known analogy pairs are enumerated in ``gold_analogy_pairs.txt`` and explained in ``gold_analogy_pairs_with-explanations.csv``.

For convenience for reviewers with Unix-like systems (e.g., Linux, OS X), the full experiment can be run from start (assuming annotations are already obtained) to finish (comparing precision of matches) by running ``run-pipeline.sh``.

# Dependencies

Note: the code in this repository is in Python 2

## Python modules
Python module dependencies (note: most of these can be obtained by simply downloading the Anaconda distribution)
* Pandas
* NLTK
* Numpy
* Scipy

## Base vectors
You'll also need gensim-formatted base vectors (will make more sense after inspecting get-vectors.py), placed in the ``models/`` subdirectory. The vectors used for Study 1 and 3 can be downloaded here: https://www.dropbox.com/s/7bqk9i2fd56xchk/glove.papers.600d.cscw%2Bchi.2010-2017.txt?dl=0.
