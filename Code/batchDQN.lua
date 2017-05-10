require 'os'
require 'nn'
require 'cunn'
require 'cunnx'
require 'optim'

--require 'cudnn'
require 'cutorch'
require 'rnn'
require 'csvigo'

dl = require 'dataload'

dofile("Code/Utils/load_cnn.lua")

outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training/'
queries, sentences, trueSummaries = loadCNN(outputpath)
vocabSize = 20001
embDim = 500

n = queries:size(1)
n_s = sentences:size(2)
k = 10

totalPredsummary = LongTensor(n, n_s * k):fill(0)

model = buildModel('bow', vocabSize, embDim, 'f1', false, false)

preds = model:forward(queries[1], sentences[1], totalPredsummary ) 
print(preds:size())
