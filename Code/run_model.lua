require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'cunnx'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext2.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'
query_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/queries_numtext.csv'
sent_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_sentence_numtext2.csv'

data_file = csvigo.load({path = aurora_fn, mode = "large"})
nugget_file = csvigo.load({path = nugget_fn, mode = "large"})
query_file =  csvigo.load({path = query_fn, mode = "large"})
sent_file =  csvigo.load({path = sent_fn, mode = "large"})

model = 'lstm'
rK = 200
batch_size = 500
nepochs = 50
print_every = 1
embed_dim = 10
learning_rate = 0.01
usecuda = true

epsilon = 1
cuts = 4.                  --- This is the number of cuts we want
base_explore_rate = 0.1
delta = 1./(nepochs/cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs

torch.manualSeed(420)

vocab_sized = getVocabSize(data_file)                       --- getting length of dictionary
vocab_sizeq = getVocabSize(query_file)                      --- getting length of dictionary
vocab_sizes = getVocabSize(sent_file)                      --- getting length of dictionary
vocab_size = math.max(vocab_sized, vocab_sizeq, vocab_sizes)

queries = grabNsamples(query_file, #query_file-1, nil)      --- Extracting all queries
nuggets = grabNsamples(nugget_file, #nugget_file-1, nil)    --- Extracting all samples
maxseqlend = getMaxseq(data_file)                             --- Extracting maximum sequence length
maxseqlenq = getMaxseq(query_file)                            --- Extracting maximum sequence length
maxseqlen = math.max(maxseqlenq, maxseqlend)

batch_model  = build_model(model, vocab_size, embed_dim, 1, usecuda)
crit = nn.MSECriterion()

out = iterateModel( batch_size, nepochs, queries[3], 
                    data_file, sent_file, 
                    batch_model, crit, epsilon, delta, 
                    maxseqlen, base_explore_rate, print_every, 
                    nuggets, learning_rate, rK, usecuda)
