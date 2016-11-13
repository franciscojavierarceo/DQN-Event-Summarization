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
cmd:option('--end_baserate', 5, 'Maximum summary size')
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
end_baserate = opt.end_baserate
n = opt.n_samples
K_tokens = 25

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
qs = inputs['query']
input_file = csvigo.load({path = data_path .. inputs['inputs'], mode = "large", verbose = false})
nugget_file = csvigo.load({path = data_path .. inputs['nuggets'], mode = "large", verbose = false})
nugget_file = geti_n(nugget_file, 2, #nugget_file) 
input_file = geti_n(input_file, 2, n) 
-- input_file = geti_n(input_file, 2, #input_file) 
local vocabSize = getVocabSize(input_file)
K_nuggs = getMaxseq(nugget_file)

nuggets = buildTermDocumentTable(nugget_file, nil)
xtdm  = buildTermDocumentTable(input_file, K_tokens)

ntdm = {}
for i=1, #nuggets do
    ntdm = tableConcat(table.unpack(nuggets), ntdm)
end

function buildEmbeddings(model, vocabSize, embeddingSize)
    if model == 'bow' then
        print(string.format("Running bag-of-words model to learn %s", metric))
        local sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.Sum(2, 3, false))
                    :add(nn.ReLU())
                    -- :add(nn.Tanh())
    else
        print(string.format("Running LSTM model to learn %s", metric))
        local sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.SplitTable(2))
                    :add(nn.Sequencer(nn.LSTM(embeddingSize, embeddingSize)))
                    :add(nn.SelectTable(-1))            -- selects last state of the LSTM
                    :add(nn.Linear(embeddingSize, embeddingSize))
                    :add(nn.ReLU())
    end
    return sentenceLookup
end

function buildModel(model, vocabSize, embeddingSize)
    local sentenceLookup = buildEmbeddings(model, vocabSize, embeddingSize)
    local queryLookup = sentenceLookup:clone("weight", "gradWeight") 
    local summaryLookup = sentenceLookup:clone("weight", "gradWeight")

    local pmodule = nn.ParallelTable()
                :add(sentenceLookup)
                :add(queryLookup)
                :add(summaryLookup)

    local nnmodel = nn.Sequential()
            :add(pmodule)
            :add(nn.JoinTable(2))
            :add(nn.ReLU())
            :add(nn.Linear(embeddingSize * 3, 2))
    return nnmodel
end

local model = buildModel(nnmod, vocabSize, embeddingSize)

local criterion = nn.MSECriterion()
local params, gradParams = model:getParameters()

if use_cuda then
    Tensor = torch.CudaTensor
    LongTensor = torch.CudaLongTensor
    ByteTensor = torch.CudaByteTensor
    criterion = criterion:cuda()
    model = model:cuda()
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

function buildMemory(newinput, memory_hist, memsize, use_cuda)
    local sentMemory = torch.cat(newinput[1][1]:double(), memory_hist[1][1]:double(), 1)
    local queryMemory = torch.cat(newinput[1][2]:double(), memory_hist[1][2]:double(), 1)
    local sumryMemory = torch.cat(newinput[1][3]:double(), memory_hist[1][3]:double(), 1)
    local rewardMemory = torch.cat(newinput[2]:double(), memory_hist[2]:double(), 1)
    local actionMemory = torch.cat(newinput[3]:double(), memory_hist[3]:double(), 1)
    --- specifying rows to index 
    if sentMemory:size(1) < memsize then
        nend = sentMemory:size(1)
        nstart = 1
    else 
        nstart = math.max(memsize - sentMemory:size(1), 1)
        nend = memsize + nstart
    end
    --- Selecting n last data points
    sentMemory = sentMemory[{{nstart, nend}}]
    queryMemory= queryMemory[{{nstart, nend}}]
    sumryMemory= sumryMemory[{{nstart, nend}}]
    rewardMemory = rewardMemory[{{nstart, nend}}]
    actionMemory = actionMemory[{{nstart, nend}}]

    local inputMemory = {sentMemory, queryMemory, sumryMemory}

    if use_cuda then
        inputMemory = {sentMemory:cuda(), queryMemory:cuda(), sumryMemory:cuda()}
    end
    return {inputMemory, rewardMemory, actionMemory}
end

function backProp(input_memory, params, model, criterion, batch_size, memsize, use_cuda)
    local inputs = {input_memory[1], input_memory[3]}
    local rewards = input_memory[2]
    local dataloader = dl.TensorLoader(inputs, rewards)
    local err = 0.    

    den = 1
    for k, xin, reward in dataloader:sampleiter(batch_size, memsize) do
        xinput = xin[1]
        actions_in = xin[2]
        local function feval(params)
            gradParams:zero()
            local predQ = model:forward(xinput)
            local maskLayer = nn.MaskedSelect()
            if use_cuda then 
                maskLayer = maskLayer:cuda()
            end
            local predQOnActions = maskLayer:forward({predQ, actions_in})

            local lossf = criterion:forward(predQOnActions, reward)
            local gradOutput = criterion:backward(predQOnActions, reward)
            local gradMaskLayer = maskLayer:backward({predQ, actions_in}, gradOutput)
            model:backward(xinput, gradMaskLayer[1])
            return lossf, gradParams
        end
        --- optim.rmsprop returns \theta, f(\theta):= loss function
        _, lossv  = optim.rmsprop(feval, params, optimParams)   
    end
    return lossv[1]
end

-- Initializing stuff
local epsilon = 1.0
local query = LongTensor{qs}
local sentenceStream = LongTensor(padZeros(xtdm, K_tokens))
local memory = {}
local refSummary = Tensor{ntdm}
local refCounts = buildTokenCounts(refSummary)
local streamSize = sentenceStream:size(1)
local buffer = Tensor(1, maxSummarySize):zero()

local perf = io.open("perf.txt", 'w')
for epoch=0, nepochs do
    actions = ByteTensor(streamSize, 2):fill(0)
    local exploreDraws = Tensor(streamSize)
    local summaryBuffer = LongTensor(streamSize + 1, maxSummarySize):zero()
    local qValues = Tensor(streamSize, 2):zero()
    rouge = Tensor(streamSize + 1):zero()

    rouge[1] = 0
    exploreDraws:uniform(0, 1)

    local summary = summaryBuffer:zero():narrow(1,1,1)
    for i=1, streamSize do
        --- the i extracts individual sentences from the stream
        local sentence = sentenceStream:narrow(1, i, 1)
        qValues[i]:copy(model:forward({sentence, query, summary}))

        if exploreDraws[i] <= epsilon then
            actions[i][torch.random(SKIP, SELECT)] = 1
        else
            if qValues[i][SKIP] > qValues[i][SELECT] then
                actions[i][SKIP] = 1
            else
                actions[i][SELECT] = 1
            end
        end

        summary = buildSummary(
            actions:narrow(1, 1, i), 
            sentenceStream:narrow(1, 1, i),
            summaryBuffer:narrow(1, i + 1, 1),
            use_cuda
            )

        local generatedCounts = buildTokenCounts(summary) 
        local recall, prec, f1 = rougeScores(generatedCounts, refCounts)

        if metric == "f1" then
            rouge[i + 1]  = f1
        elseif metric == "recall" then
            rouge[i + 1]  = recall
        elseif metric == "precision" then
            rouge[i + 1] = prec
        end

        if i==streamSize then
            rougeRecall = recall
            rougePrecision = prec
            rougeF1 = f1
        end
    end

    local max, argmax = torch.max(qValues, 2)
    local reward0 = rouge:narrow(1,2, streamSize) - rouge:narrow(1,1, streamSize)
    local reward_tp1 = gamma * reward0:narrow(1, 2, streamSize - 1):resize(streamSize)
    --- occasionally the zeros result in a nan, which is strange
    reward_tp1[reward_tp1:ne(reward_tp1)] = 0
    reward_tp1 = torch.clamp(reward_tp1, -1, 1)
    -- local reward = rouge:narrow(1,2, streamSize)
    local reward = reward0 + reward_tp1
    
    local querySize = query:size(2)
    local summaryBatch = summaryBuffer:narrow(1, 1, streamSize)
    local queryBatch = query:view(1, querySize):expand(streamSize, querySize) 

    local input = {sentenceStream, queryBatch, summaryBatch}
    --- Storing the data
    memory = {input, reward, actions}

    if epoch == 0 then
        fullmemory = memory 
    else
        local tmp = buildMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
        fullmemory = tmp
    end
    --- Running backprop
    if(epoch > n_rand) then 
        loss = backProp(memory, params, model, criterion, batch_size, mem_size, use_cuda)
    else 
        loss = 0.
    end

    if epoch==0 then
        out = string.format("epoch;epsilon;loss;rougeF1;rougeRecall;rougePrecision;actual;pred;nselect;nskip\n")
        perf:write(out)
    end
    nactions = torch.totable(actions:sum(1))[1]
    out = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}; %i; %i\n", 
        epoch, epsilon, loss, rougeF1, rougeRecall, rougePrecision,
        reward:min(), reward:max(),
        qValues:min(), qValues:max(),
        nactions[1], nactions[2]
        )
        perf:write(out)

    if export then 
        local ofile = io.open(string.format("plotdata/%s/%i_epoch.txt", nnmod, epoch), 'w')
        ofile:write("predSkip;predSelect;actual;Skip;Select\n")
        for i=1, streamSize do
            ofile:write(string.format("%.6f;%.6f;%6f;%i;%i\n", 
                    qValues[i][SKIP], qValues[i][SELECT], rouge[i], 
                    actions[i][SKIP], actions[i][SELECT]))
        end
        ofile:close()
    end 
    if (epsilon - delta) <= base_explore_rate then
        epsilon = base_explore_rate
        if epoch > end_baserate then 
            base_explore_rate = 0.
        end
    else 
        epsilon = epsilon - delta
    end
end
-- os.execute(string.format("python make_density_gif.py %i %s %s", nepochs, nnmod, metric))