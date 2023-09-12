import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import optparse
import os

def lemmatize(w):
    lemma_n = lemmatizer.lemmatize(w)
    lemma_v = lemmatizer.lemmatize(w, 'v')
    if lemma_n == lemma_v:
        return lemma_n
    else:
        if lemma_n != w:
            return lemma_n
        else:
            return lemma_v

bow_types = [
    'abstract_tokens',
    'background_tokens',
    'problem_tokens',
    'mechanism_tokens',
    'findings_tokens'
]

parser = optparse.OptionParser()
parser.add_option('--dataset', '-d', action="store", dest="dataset", help="dataset of annotations", default=None)
options, args = parser.parse_args()

in_name = os.path.join("datasets", options.dataset, "annotations_label-level_%s.p" %options.dataset)
papers = pd.read_pickle(in_name)
print papers.count()
# print papers.head()

lemmatizer = WordNetLemmatizer()

dfs_overall = {}

# let's first get the document frequencies
# we're going to try lemmas first

words_to_lemmas = {} # probably want to store the word-to-lemma mapping for later use

# per tom, we're doing this for each type of bag-of-word
# that is, we're interested in tf-idf scores for abstract, purpose, mechanism, etc...
print "Computing dfs..."
for bow_type in bow_types:
    dfs_bow = {}
    # loop through all the bags of words for this bag of word type
    for bow in papers[bow_type]:
        # convert them all to lemmas
        # and remove duplicates
        lemmas = set()
        for w in bow:
            w = w.decode('utf-8', 'ignore').encode('ascii', 'ignore') if isinstance(w, basestring) else str(w)
            lemma = lemmatize(w)
            words_to_lemmas[w] = lemma
            lemmas.add(lemma)
        # update the dfs for the lemmas in this bow
        for lemma in lemmas:
            if lemma in dfs_bow:
                dfs_bow[lemma] += 1
            else:
                dfs_bow[lemma] = 1

    # update the master dfs dict
    dfs_overall[bow_type] = dfs_bow

print "Computing idfs..."
# now that we have dfs, let's convert them to idfs
idfs_overall = {}
for bow_type, bow_dfs in dfs_overall.items():
    idfs_bow = {}
    for w, df in bow_dfs.items():
        idfs_bow[w] = np.log(len(papers)/float(df))
    idfs_overall[bow_type] = idfs_bow
idfs_overall

# now that we have idfs, let's do tf-idf for each token!
# prep the fields
for bow_type in bow_types:
    papers['tfidfs_%s' %bow_type] = ""
    papers['tfidfs_%s' %bow_type] = papers['tfidfs_%s' %bow_type].astype(object)

average_tfidf = {}
for index, row in papers.iterrows():
    for bow_type in bow_types:
        tfs_lemmas = {}
        # convert them all to lemmas and count them
        # don't remove duplicates since we're interested in frequencies
        # print "Raw bag of words for bow_type", bow_type, ":", row[bow_type]
        for w in row[bow_type]:
            w = w.decode('utf-8', 'ignore').encode('ascii', 'ignore') if isinstance(w, basestring) else str(w)
            lemma = lemmatize(w)
            if lemma in tfs_lemmas:
                tfs_lemmas[lemma] += 1
            else:
                tfs_lemmas[lemma] = 1

        tfidfs_lemmas = {}
        for l, tf in tfs_lemmas.items():
            tfidfs_lemmas[l] = float(tf)*idfs_overall[bow_type][l]
        # print "TFIDFS lemmas: ", tfidfs_lemmas
        # now map the lemma tf-idfs to the
        tfidfs_words = {}
        for word in row[bow_type]:
            word = word.decode('utf-8', 'ignore').encode('ascii', 'ignore') if isinstance(word, basestring) else str(word)
            if word not in tfidfs_words and word in words_to_lemmas:
                word_lemma = words_to_lemmas[word]
                tfidfs_words[word] = tfidfs_lemmas[word_lemma]
                if word in average_tfidf:
                    average_tfidf[word] += [tfidfs_lemmas[word_lemma]]
                else:
                    average_tfidf[word] = [tfidfs_lemmas[word_lemma]]

        # print "TFIDFS words:", tfidfs_words
        papers.set_value(index, 'tfidfs_%s' %bow_type, tfidfs_words)

# last step! make data format
papers_to_word_tfidfs = {}
for index, paper in papers.iterrows():
    # print paper['PaperID']
    paper_dict = {}
    for bow_type in bow_types:
        paper_dict[bow_type] = paper['tfidfs_%s' %bow_type]
    papers_to_word_tfidfs[paper['paperID']] = paper_dict
papers_to_word_tfidfs

out_specific = in_name.replace("label-level", "tfidfs").replace(".p", ".json").replace("annotations_", "")
open(out_specific, "w").write(json.dumps(papers_to_word_tfidfs, indent=2))
