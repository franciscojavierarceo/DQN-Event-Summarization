import os
import re
import sys
import pickle
import csv
import gzip
import numpy as np
from itertools import chain
from bs4 import BeautifulSoup
import pandas as pd
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from collections import defaultdict    

def read_queries(fname):
    f = open(fname, 'rb')
    out = f.readlines()
    ox = BeautifulSoup(''.join(out),'lxml').contents[1]
    qdata = []
    for i in ox.findAll('event'):
        qdata.append((i.findAll('query')[0].text, 
                  int(i.findAll("id")[0].text),
                  fname.split("/")[-1].replace("-ts-topics-test.xml", "").replace("trec20","TS"),
                    i.findAll('title')[0].text))
    return qdata

def gzipFile(f):
    f_in = open(f, 'rb')
    f_out = gzip.open(f + ".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()    

def loadQuery(qfilename):
    """
    :type   qfilename: str
    :param  qfilename: String indicating query file name
    """
    f = open(qfilename, 'rb')
    out = f.readlines()
    ox = BeautifulSoup(''.join(out),'lxml').contents[1]
    qs = []
    for i in ox.findAll('event'):
        qs.append(i.findAll('query')[0].text)

    return [t.split(" ") for t in qs]

def BuildIndexFiles(infile_list, qfilenames, inputdir):
    """
    :type  infile_list:  list
    :param infile_list:  List of file names to import

    :type   qfilenames:  list
    :param  qfilenames:  List of query file names
    """
    reload(sys)
    sys.setdefaultencoding('utf-8')
    all_tokens = []
    ntexts, qtexts = [], []
    frequency = defaultdict(int)
    for idx, infilename in enumerate(infile_list):        
        print('Loading and tokenizing %s (%i of %i)' % (infilename, idx+1, len(infile_list)) )

        if (infilename not in qfilenames) and ('nuggets' not in infilename):
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['text'] = df['text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().str.lower()
            texts = [t.split(" ") for t in df['text'] ]

        if 'nuggets' in infilename:
            df = pd.read_csv(infilename)
            df['nugget_text'] = df['nugget_text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip().str.lower()
            texts = [t.split(" ") for t in df['nugget_text'] ]
            ntexts.append(texts)

        if infilename in qfilenames:
            texts = loadQuery(infilename)
            qtexts.append(texts)

        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [ [token for token in text] for text in texts]
        # Collecting all the list of tokens
        all_tokens.append(texts)

    texts  = sum(all_tokens, [])
    qtexts = sum(qtexts, [])
    ntexts = sum(ntexts, [])

    # Getting the dictionary with token info
    dictionary = corpora.Dictionary(texts)
    
    # Mapping to numeric list -- adding plus one to tokens
    dictionary.token2id = {k: v+1 for k,v in dictionary.token2id.items()}
    word2idx = dictionary.token2id
    
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}
    idx2word = dictionary.id2token
    
    # Exporting the dictionaries
    print("Exporting word to index and dictionary to word indices")
    output = open(os.path.join(inputdir,'0-output/LSTMDQN_Dic_token2id.pkl'), 'ab+')
    pickle.dump(word2idx, output)
    output.close()

    output = open(os.path.join(inputdir,'0-output/LSTMDQN_Dic_id2token.pkl'), 'ab+')
    pickle.dump(idx2word, output)
    output.close()
    
    # Merging the dictionaries toa pandas data frame with summary info
    odf0 = pd.DataFrame.from_dict(dictionary.dfs, orient='index').reset_index()
    odf1 = pd.DataFrame.from_dict(word2idx, orient='index').reset_index()

    odf0.columns = ['id', 'frequency']
    odf1.columns = ['token', 'id']
    # Merge by token id
    odf = pd.merge(left=odf0, right=odf1, on='id')
    odf = odf[['id','token', 'frequency']]
    # Exporting data
    odf.to_csv(os.path.join(inputdir, '0-output/total_corpus_smry.csv'), index=False)

    return dictionary, qtexts, ntexts
    
def TokenizeData(inputdir, infile_list, qfilenames, outfile_list, word2idx, top_n, qtexts, ntexts):
    """
    :type  infile_list:  list
    :param infile_list:  List of file names to import

    :type   qfilename:    str
    :param  qfilename:    String indicating query file name

    :type  outfile_list: list
    :param outfile_list: List of file names to export, without the '.csv'

    :type  word2idx:     dic
    :param word2idx:     Dictionary of token 2 ids

    :type   top_n:      int
    :param  top_n:      Number of tokens to limit data to

    :type   qtexts:     list
    :param  qtexts:     List from the queries so they're not removed
    """
    # Loading the token frequencies
    qtexts = sum(qtexts, [])
    ntexts = sum(ntexts, [])
    tfdf = pd.read_csv(os.path.join(inputdir, '0-output/total_corpus_smry.csv'))
    tfdf['qfile'] = tfdf['token'].isin(qtexts)
    tfdf['nfile'] = tfdf['token'].isin(ntexts)
    tfdf.sort_values(by='frequency', ascending=False, inplace=True)
    tfdf = pd.concat(
                [
                    tfdf.iloc[0:top_n,:], 
                    tfdf[tfdf['qfile']==True],
                    tfdf[tfdf['nfile']==True]
                ]
            )
    tfdf.drop_duplicates(inplace=True)
    # This is the unknown text id
    maxidv = tfdf['id'].max() + 1
    tfdf = pd.concat( [tfdf, pd.Series([maxidv, 'UNKNOWN', np.nan])], axis=1)
    token_ss = dict(zip(tfdf['token'], tfdf['id']))

    for idx, (infilename, outfilename) in enumerate(zip(infile_list, outfile_list)):
        print('Loading and tokenizing %s (%i of %i)' % (infilename, idx+1, len(infile_list)) )
        if (infilename not in qfilenames) and 'nuggets' not in infilename:
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['text'] = df['text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['text'] ]

        if 'nuggets' in infilename:
            df = pd.read_csv(infilename)
            df['nugget_text'] = df['nugget_text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['nugget_text'] ]

        if infilename in qfilenames:
            texts = loadQuery(infilename)
        texts = [ [token for token in text ]  for text in texts]


        # if (qfilename not in infilename) and 'nuggets' not in infilename:
        text_numindex = [ [word2idx[i] if i in token_ss else maxidv for i in t] for t in texts]

        # Exporting files
        print('...file exported to %s.csv' % outfilename)

        with open(outfilename + '.csv', 'wb') as csvfile:
            data = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if '0-output/queries' == outfilename:
                data.writerow(['Query'])
                data.writerows(text_numindex)
            else:
                data.writerow(['Text'])
                data.writerows(text_numindex)
        csvfile.close()

    print('...Exporting of tokenized data complete')

def ExtractFirstSentence(infilelist):
    for infile in infilelist:
        tmpdf = pd.read_csv(infile, sep='\t')
        tmpdf = tmpdf[tmpdf['sentence']==0]
        newfilename = infile.replace(".tsv.gz", "_fs.tsv").replace("/archive", "")
        tmpdf.to_csv(newfilename, index=False, sep='\t')

def main(inputdir):
    if 'archive' not in os.listdir(os.path.join(inputdir, 'corpus-data')):
        os.system("mkdir %s/corpus-data/archive" % inputdir)
        os.system("mv %s/corpus-data/*.tsv.gz %s/corpus-data/archive/" % (inputdir, inputdir) )

    if 'fs' not in os.listdir(os.path.join(inputdir, 'corpus-data')):
        if len(os.listdir(os.path.join(inputdir, 'corpus-data/archive')))==0:
            os.system("mv %s/corpus-data/*.tsv.gz %s/corpus-data/archive/" % (inputdir, inputdir) )

        infilelist = os.listdir(os.path.join(inputdir, 'corpus-data/archive'))
        infilelist = [os.path.join(inputdir, 'corpus-data/archive/%s' % x) for x in infilelist if '.tsv.gz' in x]
        # Exporting the first sentence of the articles
        ExtractFirstSentence(infilelist)
        infilelist = [x for x in os.listdir(os.path.join(inputdir,'corpus-data')) if '.tsv' in x]
        # Gziping the files
        [gzipFile(os.path.join(inputdir, 'corpus-data', newfilename)) for newfilename in infilelist]
        infilelist = [x + '.gz' for x in infilelist]

    # Exporting nuggets
    nuggets = []
    nuggfiles = [os.path.join(inputdir, 'nuggets-data/nuggets_%i.tsv.gz') % x for x in range(2013, 2016)]
    for nuggfile in nuggfiles:
        tmpnuggets = pd.read_csv(nuggfile, sep='\t')
        for q in tmpnuggets['query_id'].unique():
             if "TEST" not in q:
                nuggfile = os.path.join(inputdir, "nuggets-data/%s_nuggets.csv" % q)
                tmpnuggets[tmpnuggets['query_id']==q].to_csv(nuggfile, index=False)
                nuggets.append(nuggfile)

    # First we have to segment the nuggets
    qfilenames = [os.path.join(inputdir, 'trec-data/trec%i-ts-topics-test.xml') % x for x in range(2013, 2014)]
    qtuple = list(chain(*[read_queries(xml_file) for xml_file in qfilenames ]))
    infilelist = [os.path.join(inputdir, 'corpus-data/%s_fs.tsv.gz' % t.replace(" ", "_").lower()) for (q, i, n, t)  in qtuple if i != 7]
    # Limiting the files
    input_files = [os.path.join(inputdir, 'corpus-data/', x) for x in os.listdir(os.path.join(inputdir, 'corpus-data/')) if 'tsv.gz' in x]
    infilelist = [x for x in infilelist if x in input_files]
    infilelist += qfilenames
    # Incorporating the streams
    outfilelist = [os.path.join(inputdir, '0-output/%s_tokenized' % x.split("/")[-1].split(".")[0]) for x in infilelist]
    # Incporating the nuggets
    infilelist += [os.path.join(inputdir, 'nuggets-data/%s.%i_nuggets.csv' % (n, i)) for (q, i, n, t)  in qtuple]

    infilelist = infilelist + qfilenames + nuggets
    outfilelist+= [os.path.join(inputdir, '0-output/%s.%i_nuggets_tokenized' % (n, i)) for (q, i, n, t)  in qtuple]

    # Exporting the raw files and tokenizing the data
    mycorpora, qtext, ntext = BuildIndexFiles(infilelist, qfilenames, inputdir)
    TokenizeData(inputdir = inputdir, 
                infile_list = infilelist, 
                qfilenames = qfilenames, 
                outfile_list = outfilelist, 
                word2idx = mycorpora.token2id, 
                top_n = 40000,
                qtexts = qtext,
                ntexts = ntext)
    
    # Exporting corpus summary table
    tdf = pd.read_csv(os.path.join(inputdir, "0-output/total_corpus_smry.csv"))
    tdf['stopword'] = tdf['token'].isin(STOPWORDS)
    tdfss = tdf[tdf['stopword']==True]
    tdfss['id'].to_csv(os.path.join(inputdir, "0-output/stopwordids.csv"), index=False)

    # Exporting Metadata for loading into torch
    qdf = pd.DataFrame(qtuple, columns=['query', 'query_id', 'trec', 'title'])
    qdf['nugget_file'] = qdf['trec'] + "." + qdf['query_id'].astype(str) + "_nuggets_tokenized.csv"
    qdf['stream_file'] = qdf['title'].str.replace(" ", "_").str.lower() + "_fs_tokenized.csv"
    qdf = qdf[['query_id','query','trec','nugget_file','stream_file']]
    # Adding the tokens into the file -- need to convert this to a lambda at some point
    qtokens = []
    for q in qdf["query"]:
        tokens = []
        for token in q.split(" "):
            try:
                tval = str(tdf.ix[tdf['token']==token, 'id'].values[0])
            except:
                tval = str(0)
            tokens.append(tval)
        qtokens.append(' '.join(tokens))

    qdf['tokens'] = qtokens
    qdf.to_csv(os.path.join(inputdir, "0-output/dqn_metadata.csv"), index=False)

    print("----- END ------")

if __name__ == '__main__':
    # '/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/'
    main(sys.argv[1])
