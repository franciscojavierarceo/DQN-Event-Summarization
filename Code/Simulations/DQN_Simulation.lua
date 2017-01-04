local dl = require 'dataload'
require 'nn'
require 'rnn'
require 'optim'
cmd = torch.CmdLine()

cmd:option('--nepochs', 5, 'running for 50 epochs')
cmd:option('--learning_rate', 1e-4, 'using a learning rate of 1e-5')
cmd:option('--gamma', 0., 'Discount rate parameter in backprop step')
cmd:option('--cuts', 4, 'Discount rate parameter in backprop step')
cmd:option('--base_explore_rate', 0.0, 'Base rate')
cmd:option('--n_rand', 0, 'Base rate')
cmd:option('--mem_size', 50, 'Memory size')
cmd:option('--batch_size', 10,'Batch Size')
cmd:option('--nnmod','bow','BOW/LSTM option')
cmd:option('--edim', 64,'Embedding dimension')
cmd:option('--metric', "f1", 'Metric to learn')
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
metric = opt.metric

SKIP = 1
SELECT = 2
local vocabSize = 16

local optimParams = {
    learningRate = opt.learning_rate,
}

torch.manualSeed(420)
math.randomseed(420)

if nnmod=='bow' then
    print("Running bag-of-words model")
    sentenceLookup = nn.Sequential()
                :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                :add(nn.Sum(2, 3, false))
                :add(nn.Tanh())
else
    print("Running LSTM model")
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

local function buildSummary(actions, sentences, buffer)
    buffer:zero()

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

function buildMemory(newinput, memory_hist, memsize)
    local sentMemory = torch.cat(newinput[1][1], memory_hist[1][1], 1)
    local queryMemory = torch.cat(newinput[1][2], memory_hist[1][2], 1)
    local sumryMemory = torch.cat(newinput[1][3], memory_hist[1][3], 1)
    local rewardMemory = torch.cat(newinput[2], memory_hist[2], 1)
    local actionMemory = torch.cat(newinput[3], memory_hist[3], 1)
    --- specifying rows to index 
    if sentMemory:size(1) < memsize then
        nend = sentMemory:size(1)
        nstart = 1
    else 
        nend = memsize
        nstart = math.max(memsize - sentMemory:size(1), 1)
    end
    --- Selecting n last data points
    sentMemory = sentMemory[{{n0, n}}]
    queryMemory= queryMemory[{{n0, n}}]
    sumryMemory= sumryMemory[{{n0, n}}]
    rewardMemory = rewardMemory[{{n0, n}}]
    actionMemory = actionMemory[{{n0, n}}]
    local inputMemory = {sentMemory, queryMemory, sumryMemory}
    return {inputMemory, rewardMemory, actionMemory}
end

function backProp(input_memory, params, model, criterion, batch_size, memsize)
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

local maxSummarySize = 36
local epsilon = 1.0
local query = torch.LongTensor{{0, 1, 4, 3}}
local sentenceStream = torch.Tensor{{0, 1, 3, 4}, 
                                    {7, 6, 5 ,8}, 
                                    {0, 2, 4, 3}, 
                                    {7, 5, 8, 6}, 
                                    {1, 4, 3, 2}, 
                                    {13, 14, 15, 16}}

local refSummary = torch.Tensor{{1,3,4,2,4,3,1,4,3,2,9,10,12,11}}
local refCounts = buildTokenCounts(refSummary)

local streamSize = sentenceStream:size(1)
local bestActions = torch.ByteTensor{{0,1},{1,0},{0,1},{1,0},{0,1},{1,0}}


local buffer = torch.Tensor(1, maxSummarySize):zero()
local bestSummary = buildSummary(
        bestActions:narrow(1, 1, 6), 
        sentenceStream:narrow(1, 1, 6),
        buffer:narrow(1, 1, 1)
        )

local generatedCounts = buildTokenCounts(bestSummary) 
local bestrecall, bestprec, bestf1 = rougeScores(generatedCounts, refCounts)
print(string.format("TRUE {RECALL = %.6f, PREC = %.6f, F1 = %.6f}", bestrecall, bestprec, bestf1))

local perf = io.open("sim_perf.txt", 'w')
memory = {}
for epoch=1,nepochs do
    actions = torch.ByteTensor(streamSize, 2):fill(0)
    local exploreDraws = torch.Tensor(streamSize)
    local summaryBuffer = torch.LongTensor(streamSize + 1, maxSummarySize):zero()
    local qValues = torch.Tensor(streamSize, 2):zero()
    rouge = torch.Tensor(streamSize + 1):zero()

    rouge[1] = 1
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
            summaryBuffer:narrow(1, i + 1, 1)
            )

        local generatedCounts = buildTokenCounts(summary) 
        local recall, prec, f1 = rougeScores(generatedCounts, refCounts)

        if metric == "f1" then
            rouge[i + 1] = f1
        elseif metric == "recall" then
            rouge[i + 1] = recall
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

    if epoch == 1 then
        fullmemory = memory 
    else
        local tmp = buildMemory(memory, fullmemory, mem_size, batch_size)
        fullmemory = tmp
    end
    --- Running backprop
    if(epoch > n_rand) then 
        loss = backProp(memory, params, model, criterion, batch_size, mem_size)
    else 
        loss = 0.
    end

    if epoch==1 then
        out = string.format("epoch;epsilon;loss;rougeF1;rougeRecall;rougePrecision;actual;pred\n")
        perf:write(out)
    end
     out = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}\n", 
        epoch, epsilon, loss, rougeF1, rougeRecall, rougePrecision,
        reward:min(), reward:max(),
        qValues:min(), qValues:max()
        )
        perf:write(out)
    if (epsilon - delta) <= base_explore_rate then
        epsilon = base_explore_rate
    else 
        epsilon = epsilon - delta
    end
end

print(string.format("{%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i} * Learned Actions", actions[1][1], 
    actions[1][2], 
    actions[2][1], 
    actions[2][2], 
    actions[3][1], 
    actions[3][2], 
    actions[4][1],
    actions[4][2],
    actions[5][1], 
    actions[5][2], 
    actions[6][1], 
    actions[6][2] 
    ))
print(string.format("{%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i} * Optimal Actions", 
    bestActions[1][1], 
    bestActions[1][2], 
    bestActions[2][1], 
    bestActions[2][2], 
    bestActions[3][1], 
    bestActions[3][2], 
    bestActions[4][1],
    bestActions[4][2],
    bestActions[5][1], 
    bestActions[5][2], 
    bestActions[6][1], 
    bestActions[6][2] 
    ))

print(string.format("Model rouge = %.6f; Best rouge = %.6f; Ratio = %.6f", rouge[streamSize+1], bestf1, rouge[streamSize+1]/bestf1))
-- os.execute(string.format("python plotsim.py %i %s", nepochs, nn_model))