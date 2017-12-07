import os
import sys
import pandas as pd
sys.path.append('../src')
import data_io, params, SIF_embedding
import return_chunked

wordfile = '/home/francisco/GitHub/SIF/data/glove.840B.300d.txt'   # word vector file, can be downloaded from GloVe website
weightfile = '/home/francisco/GitHub/SIF/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3   # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 0            # ignoring principal component removal

cnn_files = '/home/francisco/GitHub/cnn-dailymail/finished_files/chunked/'
cnn_path = [x for x in os.listdir(cnn_files) if 'bin' in x]


for cnn_file in cnn_files:
    abstract, sentences = show_chunked.getsentences(cnn_file)

