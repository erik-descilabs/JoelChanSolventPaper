#!/bin/bash

# datasets vary by deployment (i.e., who annotated it)
declare -a DATASETS=( "cscw50_researchers" "cscw50_upworkers" "cscw50_mturk" )

for DATASET in ${DATASETS[@]}; do

  echo item: $DATASET

  # get label-level data from word-level annotations
  python word-level-to-label-level.py \
    -d $DATASET

  # compute tf-idfs
  python compute-tf-idfs.py \
    -d $DATASET

  # get tf-idf weighted vectors
  # -v specifies base vectors to combine with tf-idf weights
  python get-vectors.py \
    -d $DATASET \
    -v models/glove.papers.600d.cscw+chi.2010-2017.txt # download this file first before running the script (see README for more details)

  # run experiment
  python use-labels-experiment-tfidf.py \
    -d $DATASET

done
