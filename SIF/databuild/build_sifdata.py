import glob
import os
import sys
import struct
import pandas as pd
from nltk.tokenize import sent_tokenize
from tensorflow.core.example import example_pb2

sys.path.append('../src')
import data_io, params, SIF_embedding

def return_bytes(reader_obj):
    len_bytes = reader_obj.read(8)
    str_len = struct.unpack('q', len_bytes)[0]
    e_s = struct.unpack("%ds" % str_len, reader_obj.read(str_len))
    es = e_s[0]
    c = example_pb2.Example.FromString(es)
    article  = str(c.features.feature['article'].bytes_list.value[0])
    abstract = str(c.features.feature['abstract'].bytes_list.value[0])
    ab = sent_tokenize(abstract)
    clean_article = sent_tokenize(article)
    clean_abstract = '. '.join([' '.join(s for s in x.split() if s.isalnum()) for x in ''.join(ab).replace("<s>","").split("</s>")]).strip()
    return clean_abstract, clean_article


def load_embed(wordfile, weightfile, weightpara=1e-3, param=None, rmpc=0):
    '''
    wordfile:   : location of embedding data (e.g., glove embedings)
    weightfile: : location of TF data for words
    weightpara: : the parameter in the SIF weighting scheme, usually in range [3e-5, 3e-3]
    rmpc:       : number of principal components to remove in SIF weighting scheme
    '''
    # input
    wordfile = '/home/francisco/GitHub/SIF/data/glove.840B.300d.txt'   # word vector file, can be downloaded from GloVe website
    weightfile = '/home/francisco/GitHub/SIF/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency

    # load word vectors
    (words, Weights) = data_io.getWordmap(wordfile)

    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word

    # set parameters
    param.rmpc = rmpc

    return Weights, words, word2weight, weight4ind

def return_sif(sentences, words, weight4ind, param, Weights):
    # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    x, m = data_io.sentences2idx(sentences, words)
    w = data_io.seq2weight(x, m, weight4ind) # get word weights
    # get SIF embedding
    embeddings = SIF_embedding.SIF_embedding(Weights, x, w, param) # embedding[i,:] is the embedding for sentence i
    return embeddings


def embed_sentences(inputpath, wordfile, weightfile, weightpara, param, rmpc, file_list):
    Weights, words, word2weight, weight4ind = load_embed(wordfile, weightfile, weightpara, param, rmpc)

    print('embeddings loaded...')
    for file_i in file_list:
        input_file = open(os.path.join(inputpath, file_i), 'rb')
        c = 0
        while input_file:
            try:
                clean_abstract, clean_article = return_bytes(input_file)
            except:
                input_file = None
            print('article cleaned...')
            embeddings = return_sif(clean_article, words, weight4ind, param, Weights)

            sdf = pd.DataFrame(clean_article, columns=['sentence'])
            sdf['clean_sentence'] = [' '.join([s for s in x if s.isalnum()]) for x in sdf['sentence'].str.split(" ")]
            sdf['summary'] = clean_abstract
            sdf.ix[1:, 'summary'] = ''

            embcols = ['emb_%i'%i for i in range(embeddings.shape[1])]
            emb = pd.DataFrame(embeddings, columns = embcols)

            sdf = pd.concat([sdf, emb], axis=1)
            sdf = sdf[['summary', 'sentence', 'clean_sentence'] + sdf.columns[3:].tolist()]
            newfile = file_i.replace(".bin", "").split("/")[-1]
            sdf.to_csv("/home/francisco/GitHub/DQN-Event-Summarization/data/sif/%s_%i.csv" % (
                     newfile, c
                     )
                )
            if (c % 100) == 0:
                 print("Data exported to %s_%i.csv" % (newfile, c))
            c+= 1

def main():
    myparams = params.params()
    mainpath = 'home/francisco/GitHub/SIF/'
    datapath = '/home/francisco/GitHub/cnn-dailymail/finished_files/chunked/'
    wordf = os.path.join(mainpath, 'data/glove.840B.300d.txt')
    weightf = os.path.join(mainpath, 'auxiliary_data/enwiki_vocab_min200.txt')
    wp = 1e-3
    rp = 0
    # Example case
    # fl = ['/home/francisco/GitHub/cnn-dailymail/finished_files/chunked/train_000.bin']
    fl = os.listdir(datapath)
    embed_sentences(datapath, wordf, weightf, wp, myparams, rp, fl)

if __name__ == "__main__":
    main()
