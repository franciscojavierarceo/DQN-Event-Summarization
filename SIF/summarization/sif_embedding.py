import sys
import pandas as pd
sys.path.append('../src')
import data_io, params, SIF_embedding
import show_chunked

# input
wordfile = '/home/francisco/GitHub/SIF/data/glove.840B.300d.txt'   # word vector file, can be downloaded from GloVe website
weightfile = '/home/francisco/GitHub/SIF/auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 0
# rmpc = 1 # they usually set 1 ; number of principal components to remove in SIF weighting scheme

# sentences = ['this is an example sentence', 'this is another sentence that is slightly longer']

cnn_file = '/home/francisco/GitHub/cnn-dailymail/finished_files/chunked/train_000.bin'
abstract, sentences = show_chunked.getsentences(cnn_file)

print(abstract)

# load word vectors
(words, We) = data_io.getWordmap(wordfile)

# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word

# load sentences
x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc

# get SIF embedding
embeddings = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i

sdf = pd.DataFrame(sentences, columns=['sentence'])
emb = pd.DataFrame(embeddings, columns=['emb_%i' % x for x in range(embeddings.shape[1])])

sdf = pd.concat([sdf, emb], axis=1)
print(sdf.head())

# for sentence, embedding in zip(sentences, embeddings):
#    print(sentence, embedding.shape, embedding.mean())
