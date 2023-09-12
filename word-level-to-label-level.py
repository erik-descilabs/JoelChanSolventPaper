import pandas as pd
import optparse
import os

parser = optparse.OptionParser()
parser.add_option('--dataset', '-d', action="store", dest="dataset", help="dataset of annotations", default=None)
options, args = parser.parse_args()

in_name = os.path.join("datasets", options.dataset, "annotations_word-level_%s.xlsx" %options.dataset)
word_level = pd.read_excel(in_name)

label_level = []
for paperID, paperData in word_level.groupby("paperID"):
    row_dict = {
        'paperID': paperID,
        'abstract_tokens': [w for w in paperData['content']],
        'background_tokens': [w for w in paperData[paperData['winningHighlight'] == "Background"]['content']],
        'problem_tokens': [w for w in paperData[paperData['winningHighlight'] == "Purpose"]['content']],
        'mechanism_tokens': [w for w in paperData[paperData['winningHighlight'] == "Mechanism"]['content']],
        'findings_tokens': [w for w in paperData[paperData['winningHighlight'] == "Findings"]['content']]
    }
    label_level.append(row_dict)
label_level = pd.DataFrame(label_level)

out_name = in_name.replace("word-level", "label-level").replace(".xlsx", ".p")
label_level.to_pickle(out_name)

out_name = in_name.replace("word-level", "label-level").replace(".xlsx", ".csv")
label_level.to_csv(out_name)
