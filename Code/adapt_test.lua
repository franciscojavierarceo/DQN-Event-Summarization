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

datapath ='data/0-output/'

cmd = torch.CmdLine()

cmd:option('--nepochs', 5, 'running for 50 epochs')
cmd:option('--learning_rate', 1e-4, 'using a learning rate of 1e-5')
cmd:option('--embeddingSize', 64,'Embedding dimension')
cmd:option('--gamma', 0., 'Discount rate parameter in backprop step')
cmd:option('--model','bow','BOW/LSTM option')
cmd:option('--cuts', 4, 'Discount rate parameter in backprop step')
cmd:option('--epsilon', 1., 'Random search rate')
cmd:option('--base_explore_rate', 0.0, 'Base rate')
cmd:option('--end_baserate', 5, 'Epoch number at which the base_rate ends')
cmd:option('--batch_size', 200,'Batch Size')
cmd:option('--mem_size', 100, 'Memory size')
cmd:option('--metric', "f1", 'Metric to learn')
cmd:option('--n_samples', 20, 'Number of samples to use')
cmd:option('--maxSummarySize', 300, 'Maximum summary size')
cmd:option('--K_tokens', 25, 'Maximum number of tokens for each sentence')
cmd:option('--thresh', 0, 'Threshold operator')
cmd:option('--adapt', false, 'Domain Adaptation Regularization')
cmd:option('--datapath', 'data/0-output/', 'Path to input data')
cmd:option('--usecuda', false, 'running on cuda')
cmd:option('--n_backprops', 3, 'Number of times to backprop through the data')
cmd:text()
--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

Tensor = torch.Tensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

use_cuda = false
thresh = 0
edim = 10
metric = 'f1'

inputs = loadMetadata(datapath .. "dqn_metadata.csv")                                               
stoplist = loadStopdata(datapath .. 'stopwordids.csv')

adapt = true

vocabSize, query_data = initialize_variables(inputs, 20, datapath, 5, 30, stoplist, thresh, use_cuda)

model = buildModel('bow', vocabSize, edim, metric, true, use_cuda)

criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.BCECriterion())

local SKIP = 1
local SELECT = 2

query_randomF1 = {}
local params, gradParams = model:getParameters()

optimParams = { learningRate = opt.learning_rate }

query_id = 1

-- local sentenceStream = query_data[query_id][1]
-- local streamSize = query_data[query_id][2]
-- local query = query_data[query_id][3]
-- local actions = query_data[query_id][4]
-- local exploreDraws = query_data[query_id][5]
-- local summaryBuffer = query_data[query_id][6]
-- local qValues = query_data[query_id][7]
-- local rouge = query_data[query_id][8]
-- local actionsOpt = query_data[query_id][9]
-- local rougeOpt = query_data[query_id][10]
-- local refSummary = query_data[query_id][11]
-- local refCounts = query_data[query_id][12]
-- local buffer = query_data[query_id][13]

-- Have to set clear the inputs at the beginning of each scoring round
-- actions:fill(0)
-- actionsOpt:fill(0)
-- rouge:fill(0)
-- rougeOpt:fill(0)
-- qValues:fill(0)
-- summaryBuffer:fill(0)
-- buffer:fill(0)
-- exploreDraws:fill(0)
-- exploreDraws:uniform(0, 1)
-- summary = summaryBuffer:zero():narrow(1, 1, 1) -- summary starts empty

-- maskLayer = nn.MaskedSelect()

-- i = 1

-- local sentence = sentenceStream:narrow(1, i, 1)

-- predTotal = model:forward({sentence, query, summary})
-- predQ = predTotal[1]
-- predReg = predTotal[2]
-- actions = ByteTensor(1, 2):fill(0)

-- if qValues[i][SKIP] > qValues[i][SELECT] then
--     actions[i][SKIP] = 1
-- else
--     actions[i][SELECT] = 1
-- end

-- local predQOnActions = maskLayer:forward({predQ[i], actions[i]}) 

-- reward = torch.zeros(1):fill(0.23):resize(1,1)
-- class = torch.ones(1):resize(1,1)

-- nll = nn.BCECriterion()
-- mse = nn.MSECriterion()
-- pc = nn.ParallelCriterion():add(mse):add(nll)

-- lossf = criterion:forward({predQOnActions, predReg}, {reward, class})

-- gradOutput = criterion:backward({predQOnActions, predReg}, {reward, class})
-- gradMaskLayer = maskLayer:backward({predQ, actions}, gradOutput[1])

-- print(model:backward({sentence, query, summary}, {gradMaskLayer[1], gradOutput[1]}) )
-- print('success')

memory, rougeRecall, rougePrecision, rougeF1, qValues = forwardpass(
                query_data, query_id, model, 1, 0., 
                metric, thresh, stoplist, adapt, use_cuda
)

loss = backProp(memory, params, gradParams, optimParams, model, criterion, 10, 20, adapt, use_cuda)
print(loss)