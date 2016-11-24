require 'optim'
require 'io'
require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'cunnx'

local dl = require 'dataload'
cmd = torch.CmdLine()

cmd:option('--nepochs', 5, 'running for 50 epochs')
cmd:option('--learning_rate', 1e-5, 'using a learning rate of 1e-5')
cmd:option('--gamma', 0.4, 'Discount rate parameter in backprop step')
cmd:option('--cuts', 4, 'Discount rate parameter in backprop step')
cmd:option('--base_explore_rate', 0.0, 'Base rate')
cmd:option('--n_rand', 0, 'Base rate')
cmd:option('--mem_size', 100, 'Memory size')
cmd:option('--batch_size', 200,'Batch Size')
cmd:option('--nnmod','bow','BOW/LSTM option')
cmd:option('--edim', 64,'Embedding dimension')
cmd:option('--usecuda', false, 'running on cuda')
cmd:option('--metric', "f1", 'Metric to learn')
cmd:option('--n_samples', 500, 'Number of samples to use')
cmd:option('--max_summary', 300, 'Maximum summary size')
cmd:text()
--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

nepochs = opt.nepochs
gamma = opt.gamma
delta = 1./(opt.nepochs/opt.cuts) 
base_explore_rate = opt.base_explore_rate
n_rand = opt.n_rand
mem_size = opt.mem_size
batch_size = opt.batch_size
nnmod = opt.nnmod
embeddingSize = opt.edim
use_cuda = opt.usecuda
metric = opt.metric
maxSummarySize = opt.max_summary
n = opt.n_samples
SKIP = 1
SELECT = 2
bow = false
export = true

local optimParams = {
    learningRate = opt.learning_rate,
}

dofile("utils.lua")
dofile("model_utils.lua")
data_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'
query_fn = data_path .. 'queries_numtext.csv'
query_file =  csvigo.load({path = query_fn, mode = "large", verbose = false})
queries = buildTermDocumentTable(query_file, nil)

torch.manualSeed(420)
math.randomseed(420)
inputs = {
        ['inputs'] = '2012_aurora_shooting_first_sentence_numtext2.csv', 
        ['nuggets'] = 'aurora_nuggets_numtext.csv',
        ['query'] = queries[3]
}
query_id = 1
K_tokens = 25
qs = inputs['query']
input_file = csvigo.load({path = data_path .. inputs['inputs'], mode = "large", verbose = false})
nugget_file = csvigo.load({path = data_path .. inputs['nuggets'], mode = "large", verbose = false})
input_file = geti_n(input_file, 2, n) 
-- input_file = geti_n(input_file, 2, #input_file) 
local vocabSize = getVocabSize(input_file)
-- nugget_file = geti_n(nugget_file, 2, #nugget_file) 
K_nuggs = getMaxseq(nugget_file)

nuggets = buildTermDocumentTable(nugget_file, nil)
xtdm  = buildTermDocumentTable(input_file, K_tokens)

ntdm = {}
for i=1, #nuggets do
    ntdm = tableConcat(table.unpack(nuggets), ntdm)
end

if nnmod=='bow' then
    print(string.format("Running bag-of-words model to learn %s", metric))
    sentenceLookup = nn.Sequential()
                :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                :add(nn.Sum(2, 3, false))
                :add(nn.Tanh())
else
    print(string.format("Running LSTM model to learn %s", metric))
    sentenceLookup = nn.Sequential()
                :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                :add(nn.SplitTable(2))
                :add(nn.Sequencer(nn.LSTM(embeddingSize, embeddingSize)))
                :add(nn.SelectTable(-1))            -- selects last state of the LSTM
                :add(nn.Linear(embeddingSize, embeddingSize))
                :add(nn.Tanh())
end
local queryLookup = sentenceLookup:clone("weight", "gradWeight") 
local summaryLookup = sentenceLookup:clone("weight", "gradWeight")

local pmodule = nn.ParallelTable()
            :add(sentenceLookup)
            :add(queryLookup)
            :add(summaryLookup)

local model = nn.Sequential()
        :add(pmodule)
        :add(nn.JoinTable(2))
        :add(nn.Tanh())
        :add(nn.Linear(embeddingSize * 3, 2))

local criterion = nn.MSECriterion()
local params, gradParams = model:getParameters()

if use_cuda then
    Tensor = torch.CudaTensor
    LongTensor = torch.CudaLongTensor
    ByteTensor = torch.CudaByteTensor
    criterion = criterion:cuda()
    print("...running on GPU")
else
    torch.setnumthreads(8)
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor
    print("...running on CPU")
end

local function buildSummary(actions, sentences, buffer, use_cuda)
    buffer:zero()

    if use_cuda then 
        actions = actions:double()
        sentences = sentences:double()
        buffer = buffer:double()
    end
    local bufferSize = buffer:size(2)
    local actionsSize = actions:size(1)
    local sentencesSize = sentences:size(2)

    local mask1 = torch.eq(actions:select(2,2), 1):view(actionsSize, 1):expand(actionsSize, sentencesSize)
    local allTokens = sentences:maskedSelect(mask1)
    local mask2 = torch.gt(allTokens,0)
    local allTokens = allTokens:maskedSelect(mask2)

    if allTokens:dim() > 0 then
        local copySize = math.min(bufferSize, allTokens:size(1))

        buffer[1]:narrow(1, bufferSize - copySize + 1, copySize):copy(
            allTokens:narrow(1, allTokens:size(1) - copySize + 1, copySize)
            )
    end
    if use_cuda then
        buffer = buffer:cuda()
    end
    return buffer
end

function buildTokenCounts(summary)
    local counts = {}
    for i=1,summary:size(2) do
        if summary[1][i] > 0 then
            local token = summary[1][i]
            if counts[token] == nil then
                counts[token] = 1
            else
                counts[token] = counts[token] + 1
            end
        end
    end
    return counts
end

function rougeScores(genSummary, refSummary)
    local genTotal = 0
    local refTotal = 0
    local intersection = 0
    for k, refCount in pairs(refSummary) do
        local genCount = genSummary[k]
        if genCount == nil then genCount = 0 end
        intersection = intersection + math.min(refCount, genCount)
        refTotal = refTotal + refCount
    end
    for k,genCount in pairs(genSummary) do
        genTotal = genTotal + genCount
    end

    if genTotal == 0 then 
        genTotal = 1 
    end
    local recall = intersection / refTotal
    local prec = intersection / genTotal
    if recall > 0 and prec > 0 then
        f1 = 2 * recall * prec / (recall + prec)
    else 
        f1 = 0
    end
    return recall, prec, f1
end

local epsilon = 1.0
local query = LongTensor{qs}
local sentenceStream = LongTensor(padZeros(xtdm, K_tokens))

refSummary = Tensor{ntdm}
refCounts = buildTokenCounts(refSummary)
streamSize = sentenceStream:size(1)
buffer = Tensor(1, maxSummarySize):zero()
sentenceStream = LongTensor(padZeros(xtdm, K_tokens))

memory = {}

actions = ByteTensor(streamSize, 2):fill(0)
summaryBuffer = LongTensor(streamSize + 1, maxSummarySize):zero()
score = 0
for i=1, streamSize do
    actions[i][SELECT] = 1
    summary = buildSummary(
        actions:narrow(1, 1, i), 
        sentenceStream:narrow(1, 1, i),
        summaryBuffer:narrow(1, i + 1, 1),
        use_cuda
        )
    local generatedCounts = buildTokenCounts(summary) 
    local recall, prec, f1 = rougeScores(generatedCounts, refCounts)
    if f1 < score then
        actions[i][SELECT] = 0
        actions[i][SKIP] = 1
    end
    if f1 > score then
        score = f1
    end
print(score)
end
print(torch.totable(actions:sum(1))[1][SELECT])