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
dofile("Code/utilsNN.lua")

outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training/'

torch.setnumthreads(8)
Tensor = torch.Tensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
print("...running on CPU")

queries, sentences, trueSummaries = loadCNN(outputpath, 10000)
vocabSize = 20001
embDim = 500

n = queries[1]:size(1)
n_s = sentences[1]:size(2)
k = 1

totalPredsummary = LongTensor(n, n_s * k):fill(0)

model = buildModel('bow', vocabSize, embDim, 'f1', false, false)

print(#queries[1], #sentences[1], #totalPredsummary)
preds = model:forward({queries[1], sentences[1], totalPredsummary}) 
print(preds:size())
