import os
import re
import sys
import pickle
import csv
import gzip
import numpy as np
from collections import Counter
from itertools import chain
from bs4 import BeautifulSoup
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict

df = pd.read_csv("/home/francisco/GitHub/DQN-Event-Summarization/data/1-output/cnn_trainingstreams.csv")

# Clean up summaries
#df['true_summary'] = df['true_summary'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")
df['sentence'] = df['Sentence'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")
df['query'] = df['query'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")

frequency = defaultdict(int)

n = df.shape[0]
div = n // 10
qtokens, stokens, tstokens = [], [], []
for i, row in df.iterrows():
    qtokens+= [row['query'].split(" ")]
    stokens+= [row['sentence'].split(" ")]
#    tstokens+= [row['true_summary'].split(" ")]
    if ((i + 1) % div) == 0:
        print("%i/%i (%i%%) complete rows." % (i + 1, n, (i + 1) / float(n) * 100 ))

# Getting the dictionary with token info
dictionary = corpora.Dictionary(stokens + qtokens + tstokens )

# Mapping to numeric list -- adding plus one to tokens
dictionary.token2id = {k: v + 1 for k,v in dictionary.token2id.items()}
dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}

inputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/1-output/"

print("Exporting word to index and dictionary to word indices")
output = open(os.path.join(inputdir,'LSTMDQN_Dic_token2id_cnn.pkl'), 'ab+')
pickle.dump(dictionary.token2id, output)
output.close()

output = open(os.path.join(inputdir,'LSTMDQN_Dic_id2token_cnn.pkl'), 'ab+')
pickle.dump(dictionary.id2token, output)
output.close()

odf0 = pd.DataFrame.from_dict(dictionary.dfs, orient='index').reset_index()
odf1 = pd.DataFrame.from_dict(dictionary.token2id, orient='index').reset_index()

odf0.columns = ['id', 'frequency']
odf1.columns = ['token', 'id']

# Merge by token id
odf = pd.merge(left=odf0, right=odf1, on='id')
odf = odf[['id','token', 'frequency']]

# Exporting data
odf.to_csv(os.path.join(inputdir, "cnn_total_corpus_smry.csv"), index=False)
