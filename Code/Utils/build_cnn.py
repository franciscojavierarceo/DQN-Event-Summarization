import os
import re
import sys
import pickle
import csv
import gzip
import dill
import numpy as np
from collections import Counter
from itertools import chain
from bs4 import BeautifulSoup
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict
from joblib import Parallel, delayed

def tokenize_cnn(inputdir, inputfile, outputdir, maxtokens=10000):
    df = pd.read_csv(os.path.join(inputdir, inputfile), nrows=1000)
    # Clean up summaries
    df['true_summary'] = df['true_summary'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")
    df['sentence'] = df['sentence'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")
    df['query'] = df['query'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().fillna("")

    frequency = defaultdict(int)

    n = df.shape[0]
    div = n // 10
    qtokens, stokens, tstokens = [], [], []
    for i, row in df.iterrows():
        qtokens+= [row['query'].split(" ")]
        stokens+= [row['sentence'].split(" ")]
        tstokens+= [row['true_summary'].split(" ")]
        if ((i + 1) % div) == 0:
            print("%i/%i (%i%%) complete rows." % (i + 1, n, (i + 1) / float(n) * 100 ))

    # Getting the dictionary with token info
    dictionary = corpora.Dictionary(stokens + qtokens + tstokens )

    # Mapping to numeric list -- adding plus one to tokens
    dictionary.token2id = {k: v + 1 for k,v in dictionary.token2id.items()}
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}

    print("Exporting word to index and dictionary to word indices")
    output = open(os.path.join(outputdir,'LSTMDQN_Dic_token2id_cnn.pkl'), 'ab+')
    pickle.dump(dictionary.token2id, output)
    output.close()
    output = open(os.path.join(outputdir,'LSTMDQN_Dic_id2token_cnn.pkl'), 'ab+')
    pickle.dump(dictionary.id2token, output)
    output.close()

    odf0 = pd.DataFrame.from_dict(dictionary.dfs, orient='index').reset_index()
    odf1 = pd.DataFrame.from_dict(dictionary.token2id, orient='index').reset_index()
    odf0.columns = ['id', 'frequency']
    odf1.columns = ['token', 'id']

    # Merge by token id
    odf = pd.merge(left=odf0, right=odf1, on='id')
    odf = odf[['id','token', 'frequency']]
    odf.sort_values(by='frequency', ascending=False, inplace=True)
    odf['cumfreq'] = odf['frequency'].cumsum()
    odf['percent'] = odf['frequency'] / odf['frequency'].sum()
    odf['cumpercent'] = odf['percent'].cumsum()
    odf.reset_index(drop=True, inplace=True)
    
    # Exporting data    
    odf.to_csv(os.path.join(outputdir, "cnn_total_corpus_smry.csv"), index=False)

    dictionary.save(os.path.join(outputdir, 'cnn_total_corpus.mm'))

    # Reducing the tokens here:
    dictionary.filter_extremes(keep_n = maxtokens)
    dictionary.save(os.path.join(outputdir, 'cnn_subset_corpus.mm'))

    print("There are unique %i tokens in the original data. Only using the %i most frequent tokens." % (odf.shape[0], maxtokens) )
    print("\tThis represents %i%% of the full set of tokens" % (odf.ix[maxtokens, 'cumpercent'] * 100 ))

    odf0 = pd.DataFrame.from_dict(dictionary.dfs, orient='index').reset_index()
    odf1 = pd.DataFrame.from_dict(dictionary.token2id, orient='index').reset_index()
    odf0.columns = ['id', 'frequency']
    odf1.columns = ['token', 'id']
    odf = pd.merge(left=odf0, right=odf1, on='id')
    odf = odf[['id','token', 'frequency']]
    odf.sort_values(by='frequency', ascending=False, inplace=True)
    odf.reset_index(drop=True, inplace=True)
    odf.to_csv(os.path.join(outputdir, "cnn_subset_corpus_smry.csv"), index=False)

    # Replacing the tokens here
    findf = df[['query_id', 'sentence_idx', 'query', 'sentence', 'true_summary']].copy()
    findf['stokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['sentence'].split(" ")]) , axis = 1)
    findf['tstokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['true_summary'].split(" ")]) , axis = 1)
    findf['qtokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['query'].split(" ")]) , axis = 1)

    min_idx, max_idx = findf['sentence_idx'].min(), findf['sentence_idx'].max()
    cols = ['qtokens', 'stokens', 'tstokens']

    # Exporting all of the files
    for idx in range(min_idx, max_idx + 1):
        findf.ix[ findf['sentence_idx'] == idx, cols].to_csv( 
                os.path.join(outputdir, 'cnn_data_sentence_%02d.csv' % idx ), 
            index=False)

def main():
    inputdir = sys.argv[1]
    outputdir = sys.argv[2]
    inputfile = sys.argv[3]
    maxtokens = sys.argv[4]

    if not inputdir or not outputdir or not inputfile:
        inputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/1-output/"
        outputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/2-output/"
        inputfile = "cnn_trainingstreams.csv"

    if not maxtokens:
        maxtokens = 10000

    tokenize_cnn(inputdir, inputfile, outputdir, maxtokens=int(maxtokens))

if __name__ == "__main__":
    main()
    # time python Code/Utils/build_cnn.py ./data2/1-output/ ./data2/2-output/ cnn_trainingstreams.csv 1000
