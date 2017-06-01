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
    output = open(os.path.join(outputdir,'LSTMDQN_Dic_token2id_cnn.pkl'), 'ab+')
    pickle.dump(dictionary.token2id, output)
    output.close()
    output = open(os.path.join(outputdir,'LSTMDQN_Dic_id2token_cnn.pkl'), 'ab+')
    pickle.dump(dictionary.id2token, output)
    output.close()

    odf0 = pd.DataFrame.from_dict(dictionary.dfs, orient='index').reset_index()
    ofindf = pd.DataFrame.from_dict(dictionary.token2id, orient='index').reset_index()
    odf0.columns = ['id', 'frequency']
    ofindf.columns = ['token', 'id']

    # Merge by token id
    odf = pd.merge(left=odf0, right=ofindf, on='id')
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
    ofindf = pd.DataFrame.from_dict(dictionary.token2id, orient='index').reset_index()
    odf0.columns = ['id', 'frequency']
    ofindf.columns = ['token', 'id']
    odf = pd.merge(left=odf0, right=ofindf, on='id')
    odf = odf[['id','token', 'frequency']]
    odf.sort_values(by='frequency', ascending=False, inplace=True)
    odf.reset_index(drop=True, inplace=True)
    odf.to_csv(os.path.join(outputdir, "cnn_subset_corpus_smry.csv"), index=False)

    # Replacing the tokens here
    findf = df[['query_id', 'sentence_idx', 'query', 'sentence', 'true_summary']].copy()
    findf['stokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['sentence'].split(" ")]) , axis = 1)
    findf['tstokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['true_summary'].split(" ")]) , axis = 1)
    findf['qtokens'] = df.apply(lambda row: ' '.join([ str(dictionary.token2id.get(word, maxtokens + 1)) for word in row['query'].split(" ")]) , axis = 1)

    findf.to_csv(
            os.path.join(outputdir, 'cnn_trainingstreams_tokenized.csv'),
            index=False
    )


def export_tokens(outputdir):
    cols = ['sentence_idx', 'query_id', 'qtokens', 'stokens', 'tstokens']
    findf = pd.read_csv(os.path.join(outputdir, 'cnn_trainingstreams_tokenized.csv'))
    qdf = findf[['query_id', 'qtokens']].groupby(['query_id', 'qtokens']).size().reset_index().rename(columns={0:'n_sentences'})
    qdf.drop(labels='n_sentences', axis=1, inplace=True)
    min_idx, max_idx = findf['sentence_idx'].min(), findf['sentence_idx'].max()
    # Exporting all of the files
    for idx in range(min_idx, max_idx + 1):
        findf_ssidx = findf[findf['sentence_idx'] == idx].copy()
        findf_ssidx.drop_duplicates(inplace=True)
        if idx == 0 :
            qdfout = qdf.merge(findf_ssidx[['query_id', 'stokens']], 
                how='left', on=['query_id']
            ) 
        else:
            print(idx, idx + 3)
            qdfout = qdfout.merge(findf_ssidx[['query_id', 'stokens']], 
                how='left', on=['query_id']
            ) 
            
        qdfout.columns = qdfout.columns[:(3 + idx) ].tolist() + ['stokens_%i' % idx]
        print(qdfout.head())

    qdfout.to_csv(
            os.path.join(outputdir, 'cnn_data.csv'), 
        index=False
    )

def export_tokens_ss(inputdir, outputdir):
    cols = ['sentence_idx', 'query_id', 'qtokens', 'stokens', 'tstokens']
    findf = pd.read_csv(os.path.join(inputdir, 'cnn_trainingstreams_tokenized.csv'))
    # chose 67 because it represents 
    findf['slen'] = findf.apply(lambda row: len(row['stokens'].split(" ")), axis=1)
    sdf = pd.concat([findf['slen'].value_counts(), findf['slen'].value_counts(normalize=True)], axis=1).reset_index()
    sdf.columns = ['sent_len', 'count', 'percent']
    sdf.sort_values(by='sent_len', inplace=True)
    sdf.reset_index(inplace=True, drop=True)
    sdf['cumpercent'] = sdf['percent'].cumsum()
    xval = sdf[sdf['cumpercent'] <=0.99].shape[0]
    findf.drop('slen', inplace=True, axis =1) 

    findf['stokens'] = findf.apply(lambda row: ' '.join(row['stokens'].split(" ")[0:xval]) , axis=1)

    qdf = findf[['query_id', 'qtokens']].groupby(['query_id', 'qtokens']).size().reset_index().rename(columns={0:'n_sentences'})
    qdf.drop(labels='n_sentences', axis=1, inplace=True)
    min_idx, max_idx = findf['sentence_idx'].min(), findf['sentence_idx'].max()
    # Exporting all of the files
    for idx in range(min_idx, max_idx + 1):
        findf_ssidx = findf[findf['sentence_idx'] == idx].copy()
        findf_ssidx.drop_duplicates(inplace=True)
        if idx == 0 :
            qdfout = qdfm.merge(findf_ssidx[['query_id', 'stokens']], 
                how='left', on=['query_id']
            ) 
        else:
            qdfout = qdfout.merge(findf_ssidx[['query_id', 'stokens']], 
                how='left', on=['query_id']
            ) 
            
        qdfout.columns = qdfout.columns[:(3 + idx) ].tolist() + ['stokens_%i' % idx]
        print(qdfout.head())

    qdfout.to_csv(
            os.path.join(outputdir, 'cnn_data_ss.csv'), 
        index=False
    )
def main():
    inputdir = sys.argv[1]
    outputdir = sys.argv[2]
    inputfile = sys.argv[3]
    maxtokens = sys.argv[4]
    outputdirss = sys.argv[5]

    if not inputdir or not outputdir or not inputfile:
        inputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/1-output/"
        outputdir = "/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized/"
        outputdirss = "/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized_ss/"
        inputfile = "cnn_trainingstreams.csv"

    if not maxtokens:
        maxtokens = 10000

    if not 'cnn_trainingstreams_tokenized.csv' in os.listdir(outputdir):
        print("running tokenization...")
        tokenize_cnn(inputdir, inputfile, outputdir, maxtokens=int(maxtokens))

    print("exporting tokens...")
    export_tokens(outputdir)
    export_tokens_ss(outputdir, outputdirss)
    print("...processing complete")

if __name__ == "__main__":
    main()
    # time python Code/Utils/build_cnn.py ./data2/1-output/ ./data2/2-output/ cnn_trainingstreams.csv 1000
    # time python Code/Utils/build_cnn.py ./data/1-output/ ./data/cnn_tokenized/ cnn_trainingstreams.csv 20000 ./data/cnn_tokenized_ss/
