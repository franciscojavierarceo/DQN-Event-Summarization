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

def returntoken(corpus, word, maxtokens):
    return corpus[word] if word in corpus else maxtokens + 1 

def tokenize_cnn(inputdir, inputfile, outputpath, maxtokens=10000):
    df = pd.read_csv(os.path.join(inputdir, inputfile))
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
    output = open(os.path.join(outputpath,'LSTMDQN_Dic_token2id_cnn.pkl'), 'ab+')
    pickle.dump(dictionary.token2id, output)
    output.close()
    output = open(os.path.join(outputpath,'LSTMDQN_Dic_id2token_cnn.pkl'), 'ab+')
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
    odf.to_csv(os.path.join(outputpath, "cnn_total_corpus_smry.csv"), index=False)

    # Reducing the tokens here:
    corpus = dictionary.filter_extremes(keep_n = maxtokens)

    print("There are unique %i tokens in the original data. Only using the %i most frequent tokens." % (odf.shape[0], maxtokens) )
    print("\tThis represents %i%% of the full set of tokens" % (odf.ix[maxtokens, 'cumpercent'] * 100 ))

    odf0 = pd.DataFrame.from_dict(corpus.dfs, orient='index').reset_index()
    odf1 = pd.DataFrame.from_dict(corpus.token2id, orient='index').reset_index()
    odf0.columns = ['id', 'frequency']
    odf1.columns = ['token', 'id']
    odf = pd.merge(left=odf0, right=odf1, on='id')
    odf = odf[['id','token', 'frequency']]
    odf.sort_values(by='frequency', ascending=False, inplace=True)
    odf.reset_index(drop=True, inplace=True)
    odf.to_csv(os.path.join(outputpath, "cnn_subset_corpus_smry.csv"), index=False)

    # Replacing the tokens here
    findf = df[['query_id', 'sentence_idx', 'query', 'sentence', 'true_summary']].copy()
    findf['stokens'] = [ ' '.join([ str(returntoken(corpus.token2id, word, maxtokens)) for word in row ]) for row in findf['sentence'].str.split(" ")]
    findf['tstokens'] = [ ' '.join([ str(returntoken(corpus.token2id, word, maxtokens)) for word in row ]) for row in findf['true_summary'].str.split(" ")]
    findf['qtokens'] = [ ' '.join([ str(returntoken(corpus.token2id, word, maxtokens)) for word in row ]) for row in findf['query'].str.split(" ")]
    min_idx, max_idx = findf['sentence_idx'].min(), findf['sentence_idx'].max()

    cols = ['qtokens', 'stokens', 'tstokens']
    # Exporting all of the files
    for idx in range(min_idx, max_idx + 1):
        findf.ix[ findf['sentence_idx'] == idx, cols].to_csv( 
                os.path.join(outputpath, 'cnn_data_sentence_%02d.csv' % idx.zfill(2)), 
            index=False)

def main():
    inputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/1-output/"
    inputfile = "cnn_trainingstreams.csv"
    outputpath = "/home/francisco/GitHub/DQN-Event-Summarization/data/2-output/"
    tokenize_cnn(inputdir, inputfile, outputpath, maxtokens=10000)

if __name__ == "__main__":
    main()
