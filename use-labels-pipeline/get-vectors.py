import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import json
import optparse
from operator import itemgetter
from operator import attrgetter
import os

"""
0. Define parameters
"""
# define command-line args
parser = optparse.OptionParser()
parser.add_option('--dataset', '-d', action="store", dest="dataset", help="dataset of annotations", default=None)
parser.add_option('--vectors', '-v', action="store", dest="vectors", help="base vectors to use", default=None)
options, args = parser.parse_args()

# define vector types
BOW_TYPES = {
    'abstract_tokens': ['abstract_tokens'],
    'problem_tokens': ['problem_tokens'],
    'big_problem_tokens': ['background_tokens', 'problem_tokens'],
    'mechanism_tokens': ['mechanism_tokens'],
    'findings_tokens': ['findings_tokens'],
    'background_tokens': ['background_tokens'],
}

TFIDF_CUTOFF = None

"""
1. Read in the data
"""

# read in label-level annotations
papers_in = os.path.join("datasets", options.dataset, "annotations_label-level_%s.p" %options.dataset)
PAPERS = pd.read_pickle(papers_in)

# read in base vectors
BASE_VECTOR_FILE = options.vectors
BASE_VECTORS = {}
for line in open(BASE_VECTOR_FILE).readlines():
    lineData = line.split(" ")
    BASE_VECTORS[lineData[0]] = np.array([float(d) for d in lineData[1:]])

# read in tfidf weights
tfidf_in = os.path.join("datasets", options.dataset, "tfidfs_%s.json" %options.dataset)
TFIDF = json.loads(open(tfidf_in).read())

"""
2. Compute vectors
"""

PAPER_VECTORS = {}
# for each paper
for index, paper in PAPERS.iterrows():
    print "processing paper %s..." %paper['paperID']
    paper_dict = {}

    # for each vector type
    for bow_type, bow_filters in BOW_TYPES.items():

        word_vectors = []
        new_word_vectors = []

        # get each annotation type that the vector is composed of
        for bow_filter in bow_filters:

            # get the tokens assigned to that annotation type for this paper
            bow = []
            for w in paper[bow_filter]:
                # deal with unicode issues
                if isinstance(w, basestring):
                    bow.append(w.decode('utf-8', 'ignore').encode('ascii', 'ignore'))
                else:
                    bow.append(str(w))

            # weight the token's base vector with its annotation-specific tfidf weight
            for w in bow:
                if w in BASE_VECTORS and w in TFIDF[paper['paperID']][bow_filter]:
                    tfidf = TFIDF[paper['paperID']][bow_filter][w]
                    word_vector = np.array(BASE_VECTORS[w])
                    word_vectors.append((w, word_vector.dot(tfidf)))

        # create overall vector by averaging across weighted token-vectors
        # and add to data structure for paper
        if len(word_vectors) > 0:
            avg_vector = np.mean(map(itemgetter(1), word_vectors), 0)
            paper_dict[bow_type] = list(avg_vector)

    # store vector data for paper
    PAPER_VECTORS[paper['paperID']] = paper_dict

# output the vectors to disk
out = tfidf_in.replace("tfidfs", "vectors")
open(out, "w").write(json.dumps(PAPER_VECTORS, indent=2))
