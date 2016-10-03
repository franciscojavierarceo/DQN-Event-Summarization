import os
import re
import sys
import pickle
import csv
from bs4 import BeautifulSoup
import pandas as pd
from gensim import corpora
from collections import defaultdict    

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

def BuildIndexFiles(infile_list, qfilename):
    """
    :type  infile_list:  list
    :param infile_list:  List of file names to import

    :type   qfilename:    str
    :param  qfilename:    String indicating query file name
    """
    reload(sys)
    sys.setdefaultencoding('utf-8')
    all_tokens = []
    frequency = defaultdict(int)
    for idx, infilename in enumerate(infile_list):        
        print('Loading and tokenizing %s (%i of %i)' % (infilename, idx+1, len(infile_list)) )
        if (qfilename not in infilename) and 'nuggets' not in infilename:
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['text'] = df['text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['text'] ]

        if 'nuggets' in infilename:
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['nugget_text'] = df['nugget_text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['nugget_text'] ]

        if infilename == qfilename:
            texts = loadQuery(infilename)
            qtexts =  texts

        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [ [token for token in text ]  for text in texts]
        # Collecting all the list of tokens
        all_tokens.append(texts)

    if qtexts == None:
        qtexts = []

    texts = sum(all_tokens, [])
    qtexts = sum(qtexts, [])

    # Getting the dictionary with token info
    dictionary = corpora.Dictionary(texts)
    
    # Mapping to numeric list -- adding plus one to tokens
    dictionary.token2id = {k: v+1 for k,v in dictionary.token2id.items()}
    word2idx = dictionary.token2id
    
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}
    idx2word = dictionary.id2token
    
    # Exporting the dictionaries
    print("Exporting word to index and dictionary to word indices")
    output = open('./0-output/LSTMDQN_Dic_token2id.pkl', 'ab+')
    pickle.dump(word2idx, output)
    output.close()

    output = open('./0-output/LSTMDQN_Dic_id2token.pkl', 'ab+')
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
    odf.to_csv('./0-output/total_corpus_smry.csv', index=False)

    return dictionary, qtexts
    
def TokenizeData(infile_list, qfilename, outfile_list, word2idx, top_n, qtexts):
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

    if qtexts == None:
        qtexts = []
    # Loading the token frequencies
    tfdf = pd.read_csv('./0-output/total_corpus_smry.csv')
    tfdf['qfile'] = tfdf['token'].isin(qtexts)
    tfdf.sort_values(by='frequency', ascending=False, inplace=True)
    tfdf = pd.concat([tfdf.iloc[0:top_n,:], tfdf[tfdf['qfile']==True]])
    tfdf.drop_duplicates(inplace=True)
    token_ss = dict(zip(tfdf['token'], tfdf['id']))

    for idx, (infilename, outfilename) in enumerate(zip(infile_list, outfile_list)):
        print('Loading and tokenizing %s (%i of %i)' % (infilename, idx+1, len(infile_list)) )
        if (qfilename not in infilename) and 'nuggets' not in infilename:
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['text'] = df['text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['text'] ]

        if 'nuggets' in infilename:
            df = pd.read_csv(infilename, sep='\t', encoding='latin-1')
            df['nugget_text'] = df['nugget_text'].str.replace('[^A-Za-z0-9]+', ' ').str.strip()
            texts = [ t.split(" ") for t in df['nugget_text'] ]

        if qfilename in infilename:
            texts = loadQuery(infilename)
        texts = [ [token for token in text ]  for text in texts]


        # if (qfilename not in infilename) and 'nuggets' not in infilename:
        text_numindex = [ [word2idx[i] for i in t if i in token_ss] for t in texts]

        # Exporting files
        print('...file exported to %s.csv' % outfilename+ '_numtext.csv')

        with open(outfilename + '_numtext.csv', 'wb') as csvfile:
            data = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if './0-output/queries' == outfilename:
                data.writerow(['Query'])
                data.writerows(text_numindex)
            else:
                data.writerow(['Text'])
                data.writerows(text_numindex)
        csvfile.close()

    print('...Exporting of tokenized data complete')

if __name__ == '__main__':
    os.chdir('/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/')
    # Original
    infilelist = [
            './corpus-data/2012_aurora_shooting.tsv.gz', 
            './corpus-data/2012_pakistan_garment_factory_fires.tsv.gz',
            './corpus-data/hurricane_sandy.tsv.gz',
            './corpus-data/wisconsin_sikh_temple_shooting.tsv.gz',
            './trec-2013-data/nuggets.tsv.gz',
            './trec-2013-data/trec2013-ts-topics-test.xml'
    ]
    outfilelist = [
            './0-output/2012_aurora_shooting',
            './0-output/2012_pakistan_garment_factory_fires',
            './0-output/hurricane_sandy',
            './0-output/wisconsin_sikh_temple_shooting',
            './0-output/nuggets',
            './0-output/queries'
    ]

    qfilename = './trec-2013-data/trec2013-ts-topics-test.xml'
    # Exporting the raw files
    mycorpora, qtext = BuildIndexFiles(infilelist, qfilename)
    TokenizeData(infile_list = infilelist, 
                qfilename = qfilename, 
                outfile_list = outfilelist, 
                word2idx = mycorpora.token2id, 
                top_n = 10000,
                qtexts = qtext)
    
    # Exporting the first setence files -- corpora based on the total list
    infilelist_fs = [
            './corpus-data/2012_aurora_shooting_first_sentence.tsv.gz', 
            './corpus-data/2012_pakistan_garment_factory_fires_first_sentence.tsv.gz',
            './corpus-data/hurricane_sandy_first_sentence.tsv.gz',
            './corpus-data/wisconsin_sikh_temple_shooting_first_sentence.tsv.gz',
            './trec-2013-data/nuggets.tsv.gz',
    ]
    outfilelist_fs = [
            './0-output/2012_aurora_shooting_first_sentence',
            './0-output/2012_pakistan_garment_factory_fires_first_sentence',
            './0-output/hurricane_sandy_first_sentence',
            './0-output/wisconsin_sikh_temple_shooting_first_sentence',
            './0-output/nuggets_first_sentence',
    ]
    TokenizeData(infile_list = infilelist_fs, 
                qfilename = qfilename, 
                outfile_list = outfilelist_fs, 
                word2idx = mycorpora.token2id, 
                top_n = 10000,
                qtexts = qtext)

    # Exporting the nuggets only -- corpora based on the total list
    infile_nuggets = [
            './trec-2013-data/aurora_nuggets.tsv.gz',
            './trec-2013-data/pakistan_nuggets.tsv.gz',
            './trec-2013-data/sandy_nuggets.tsv.gz',
            './trec-2013-data/wisconsin_nuggets.tsv.gz'
    ]
    outfile_nuggets = [
            './0-output/aurora_nuggets',
            './0-output/pakistan_nuggets',
            './0-output/sandy_nuggets',
            './0-output/wisconsin_nuggets'
    ]
    TokenizeData(infile_list = infile_nuggets, 
                qfilename = qfilename, 
                outfile_list = outfile_nuggets, 
                word2idx = mycorpora.token2id, 
                top_n = 10000,
                qtexts = qtext)

    print("----- END ------")