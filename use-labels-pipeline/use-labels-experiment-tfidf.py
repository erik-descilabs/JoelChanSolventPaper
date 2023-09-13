# basics
import pandas as pd
import numpy as np
# for matching
from scipy import spatial
# experiment utilities
import itertools as it
import json
import optparse
import os

parser = optparse.OptionParser()
parser.add_option('--dataset', '-d', action="store", dest="dataset", help="dataset of annotations", default=None)
options, args = parser.parse_args()

"""
Data input files and output paths
"""

papers_in = os.path.join("datasets", options.dataset, "vectors_%s.json" %options.dataset)
PAPERS = json.loads(open(papers_in).read())
GOLD_PAIRS = set([g.replace("\n","") for g in open("gold_analogy_pairs.txt").readlines()])

result_combos_dir = os.path.join("experiment-results/combinations", options.dataset)
if not os.path.exists(result_combos_dir):
    os.makedirs(result_combos_dir)

"""
Parameters
"""

# define range of K to explore
K = [1, 2, 5, 10, 15, 20, 25, 30]

# define similarity metrics
SIMILARITY_METRICS = {
    'ALL_WORDS': [
        ('abstract_tokens','abstract_tokens'),
    ],
    'PURP_MECH': [
        ('problem_tokens','problem_tokens'),
        ('mechanism_tokens','mechanism_tokens'),
    ],
    'BKGD_PURP_MECH': [
        ('big_problem_tokens','big_problem_tokens'),
        ('mechanism_tokens','mechanism_tokens'),
    ],
    'PURP': [
        ('problem_tokens','problem_tokens'),
    ],
    'MECH': [
        ('mechanism_tokens','mechanism_tokens'),
    ],
    'FIND': [
        ('findings_tokens', 'findings_tokens')
    ],
}

"""
Summary metric
"""

def measure_precision(predictions, gold):
    """Compute precision of predictions

    Args:
        predictions (set): Set of predicted matching pairs
        gold (set): Set of actual matching pairs

    Returns:
        precision (float): Precision score
        true_p (set): True positives
        false_p (set): False positives

    """

    true_p = gold.intersection(predictions)
    false_p = predictions.difference(true_p)
    precision = float(len(true_p))/len(predictions)

    return precision, true_p, false_p

#####################################
# prepare dataframe to store the results
#####################################
results_summarized_precision = []

#####################################
# run the experiment!
#####################################
# for each metric type
for metric, metricComponents in SIMILARITY_METRICS.items():
    print "Running similarity metric: ", metric

    print "\tGetting combinations..."
    combinations = [] # to store the raw pairs. will become a dataframe

    #####################################
    # score the pairs
    #####################################

    # for each paper combination
    for doc1, doc2 in it.combinations(PAPERS.keys(), 2):

        # except mirror combinations
        if doc1 != doc2:

            # store component pair similarities
            sims = []

            # for each component pair in the metric
            for component1, component2 in metricComponents:
                # only compute the similarity metric if both papers
                # have that component
                if component1 in PAPERS[doc1] and component2 in PAPERS[doc2]:
                    # get the respective component vectors for each paper
                    vec1 = PAPERS[doc1][component1]
                    vec2 = PAPERS[doc2][component2]
                    # compute cosine similarity between their vectors
                    sim = 1-spatial.distance.cosine(vec1, vec2)
                    sims.append(sim)

            # store the combination and its similarity
            combinations.append({
                'combination': " TO ".join(sorted([doc1, doc2])),
                'sim': np.nanmean(sims) # average of the component pair similarities
            })

    # all done iterating through the paper combinations
    # now knit together the pair data for this run
    combinations = pd.DataFrame(combinations)
    # and sort in descending order of similarity
    combinations = combinations.sort_values(by="sim", ascending=False)

    # store the pair data
    combinations_out = os.path.join(result_combos_dir, "combinations_%s_%s.csv" %(options.dataset, metric))
    combinations.to_csv(combinations_out)

    #####################################
    # now we calculate precision
    #####################################
    summary_dict_precision = {
        'metric': metric
    }
    # grab results for each setting of K
    print "\tGetting results"
    for k in K:
        print "\t\t for K = ", k

        # get top K most similar combinations as predicted matches
        setSize = int((k/100.0)*1225)
        matchData = combinations.iloc[:setSize]

        # compute precision
        matches = set(matchData['combination'].values)
        precision, true_p, false_p = measure_precision(matches, GOLD_PAIRS)
        summary_dict_precision[k] = precision

    # add the results for this metric to results summary
    results_summarized_precision.append(summary_dict_precision)

# knit the result summary dataframes and finish!
results_summarized_precision = pd.DataFrame(results_summarized_precision)
results_summarized_precision = results_summarized_precision[['metric', 1, 2, 5, 10, 15, 20, 25, 30]]
results_summarized_precision.sort_values(by='metric', inplace=True)

# store the results to disk
precision_out = os.path.join("experiment-results", "summary_precision_%s.csv" %options.dataset)
results_summarized_precision.to_csv(precision_out)

print "Done and done!"
