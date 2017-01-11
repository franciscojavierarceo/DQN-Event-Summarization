require 'optim'
require 'io'
require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'cunnx'

dl = require 'dataload'
cmd = torch.CmdLine()
dofile("Code/utils.lua")
dofile("Code/utilsNN.lua")

datapath = 'data/0-output/'

Tensor = torch.Tensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

adapt = true
use_cuda = false

thresh = 0
edim = 10
metric = 'f1'
inputs = loadMetadata(datapath .. "dqn_metadata.csv")                                               
stoplist = loadStopdata(datapath .. 'stopwordids.csv')
metric ='f1'
use_cuda = false
embeddingSize = 20

nepochs = 2
gamma= 0.
learning_rate = 0.1
epsilon = 1
stopwordlist = stopwords 
mem_size = 100
optimParams = { learningRate = learning_rate }

vocabSize, query_data = initialize_variables(inputs, 20, datapath, 5, 30, stoplist, thresh, use_cuda)

print("...query loaded")
model = buildModel(model, vocabSize, embeddingSize, metric, adapt, usecuda)

maskLayer = nn.MaskedSelect()

SKIP = 1
SELECT = 2

math.randomseed(420)
torch.manualSeed(420)
criterion = nn.MSECriterion()

if adapt then 
    criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.BCECriterion())
end


if use_cuda then
    criterion = criterion:cuda()
    model = model:cuda()
end
params, gradParams = model:getParameters()

query_id = 1 
memory, rougeRecall, rougePrecision, rougeF1, qValues = forwardpass(
                query_data, query_id, model, epsilon, gamma, 
                metric, thresh, stopwordlist, adapt, use_cuda
)
fullmemory = memory
-- Stacking data
fullmemory = stackMemory(memory, fullmemory, mem_size, adapt, use_cuda)

-- xinput = fullmemory[1]
-- reward = fullmemory[2]
-- actions_in = fullmemory[3]

xinput = memory[1]
actions_in = memory[3]
reward = memory[2]:resize(20, 1)

predTotal = model:forward(xinput)
predQ = predTotal[1]
predReg = predTotal[2]


predQOnActions = maskLayer:forward({predQ, actions_in})

ones = torch.ones(predQ:size(1))

-- ones = torch.ones(predQ:size(1)):resize(predQ:size(1), 1)
lossf = criterion:forward({predQOnActions, predReg}, {reward, ones})

gradOutput = criterion:backward({predQOnActions, predReg}, {reward, ones})
gradMaskLayer = maskLayer:backward({predQ, actions_in}, gradOutput[1] )

print({xinput, gradMaskLayer, gradOutput})

print(model:backward(xinput, {gradMaskLayer[1], gradOutput[1]} ))