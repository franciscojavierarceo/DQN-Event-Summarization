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

Tensor = torch.Tensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

use_cuda = false
thresh = 0
edim = 10
metric = 'f1'
inputs = loadMetadata(datapath .. "dqn_metadata.csv")                                               
stoplist = loadStopdata(datapath .. 'stopwordids.csv')

vocabSize, query_data = initialize_variables(inputs, 20, datapath, 5, 30, stoplist, thresh, use_cuda)

model = buildModel('bow', vocabSize, edim, metric, true, use_cuda)

if regmodel then 
    criterion = nn.ParallelCriterion()
                    :add(nn.MSECriterion(), 0.5)
                    :add(nn.ClassNLLCriterion())
end

local SKIP = 1
local SELECT = 2

if use_cuda then
    criterion = criterion:cuda()
    model = model:cuda()
end

query_randomF1 = {}
local params, gradParams = model:getParameters()

query_id = 1


local sentenceStream = query_data[query_id][1]
local streamSize = query_data[query_id][2]
local query = query_data[query_id][3]
local actions = query_data[query_id][4]
local exploreDraws = query_data[query_id][5]
local summaryBuffer = query_data[query_id][6]
local qValues = query_data[query_id][7]
local rouge = query_data[query_id][8]
local actionsOpt = query_data[query_id][9]
local rougeOpt = query_data[query_id][10]
local refSummary = query_data[query_id][11]
local refCounts = query_data[query_id][12]
local buffer = query_data[query_id][13]

-- Have to set clear the inputs at the beginning of each scoring round
actions:fill(0)
actionsOpt:fill(0)
rouge:fill(0)
rougeOpt:fill(0)
qValues:fill(0)
summaryBuffer:fill(0)
buffer:fill(0)
exploreDraws:fill(0)
exploreDraws:uniform(0, 1)
summary = summaryBuffer:zero():narrow(1, 1, 1) -- summary starts empty

i = 1
local sentence = sentenceStream:narrow(1, i, 1)

print(model:forward({sentence, query, summary}))

-- memory, rougeRecall, rougePrecision, rougeF1, qValues = forwardpass(
--                 query_data, query_id, model, 1, 0., 
--                 metric, thresh, stoplist, use_cuda
-- )


