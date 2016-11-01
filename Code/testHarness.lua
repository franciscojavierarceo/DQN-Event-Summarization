require 'nn'
require 'rnn'
require 'optim'

SKIP = 1
SELECT = 2

local vocabSize = 16
local embeddingSize = 64
local gamma = .0


local sentenceLookup = nn.Sequential():add(
    nn.LookupTableMaskZero(vocabSize, embeddingSize)):add(
    nn.Sum(2, 3, false))
local queryLookup = sentenceLookup:clone() --"weight", "gradWeight")
local summaryLookup = sentenceLookup:clone() --"weight", "gradWeight")
local model = nn.Sequential():add(
    nn.ParallelTable():add(
        sentenceLookup):add(
        queryLookup):add(
        summaryLookup)):add(
    nn.JoinTable(2)):add(
    nn.Tanh()):add(
    nn.Linear(embeddingSize * 3, 2)) --:add(
    --nn.Tanh()):add(
    --nn.Linear(embeddingSize, 2))
local criterion = nn.MSECriterion()

local params, gradParams = model:getParameters()


local function buildSummary(actions, sentences, buffer)

    buffer:zero()

    local bufferSize = buffer:size(2)
    local actionsSize = actions:size(1)
    local sentencesSize = sentences:size(2)

    local mask1 = torch.eq(actions:select(2,2), 1):view(actionsSize, 1):expand(
        actionsSize, sentencesSize)
    local allTokens = sentences:maskedSelect(mask1)
    local mask2 = torch.gt(allTokens,0)
    local allTokens = allTokens:maskedSelect(mask2)

    if allTokens:dim() > 0 then
        local copySize = math.min(bufferSize, allTokens:size(1))

        buffer[1]:narrow(1, bufferSize - copySize + 1, copySize):copy(
            allTokens:narrow(1, allTokens:size(1) - copySize + 1, copySize))
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
    for k,refCount in pairs(refSummary) do
        local genCount = genSummary[k]
        if genCount == nil then genCount = 0 end
        intersection = intersection + math.min(refCount, genCount)
        refTotal = refTotal + refCount
    end
    for k,genCount in pairs(genSummary) do
        genTotal = genTotal + genCount
    end

    if genTotal == 0 then genTotal = 1 end
    local recall = intersection / refTotal
    local prec = intersection / genTotal
    local f1 = 0 
    if recall > 0 and prec > 0 then
        f1 = 2 * recall * prec / (recall + prec)
    end
    return recall, prec, f1

end



local optimParams = {
    learningRate = 1e-5,
}

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


local buffer = torch.Tensor(1,maxSummarySize):zero()
local bestSummary = buildSummary(
        bestActions:narrow(1, 1, 6), 
        sentenceStream:narrow(1, 1, 6),
        buffer:narrow(1, 1, 1))


local generatedCounts = buildTokenCounts(bestSummary) 
local recall, prec, f1 = rougeScores(generatedCounts, refCounts)
print("BEST POSSIBLE RECALL, PREC, F1")
print(recall, prec, f1)


for epoch=1,1500 do
    local actions = torch.ByteTensor(streamSize,2):fill(0)
    local exploreDraws = torch.Tensor(streamSize)
    local summaryBuffer = torch.LongTensor(
        streamSize + 1, maxSummarySize):zero()
    local qValues = torch.Tensor(streamSize, 2):zero()
    local rouge = torch.Tensor(streamSize + 1):zero()

    rouge[1] = 1
    exploreDraws:uniform(0, 1)

    local summary = summaryBuffer:zero():narrow(1,1,1)


    for i=1,streamSize do
       

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
            summaryBuffer:narrow(1, i + 1, 1))

        local generatedCounts = buildTokenCounts(summary) 
        local recall, prec, f1 = rougeScores(generatedCounts, refCounts)
        rouge[i + 1] = f1

    end

    local max, argmax = torch.max(qValues, 2)
    local reward = rouge:narrow(1,2, streamSize) 
        - rouge:narrow(1,1, streamSize)
        
    reward:narrow(1, 1, streamSize - 1):add(
        torch.mul(max, gamma):narrow(1, 2, streamSize - 1))

    --local reward:narrow(1, 1, streamSize - 1):add(
    --    torch.mul(reward:narrow(1, 2, streamSize - 1), gamma))


    local querySize = query:size(2)
    local summaryBatch = summaryBuffer:narrow(1, 1, streamSize)
    local queryBatch = query:view(1, querySize):expand(streamSize, querySize) 

    local input = {sentenceStream, queryBatch, summaryBatch}

    local function feval(params)
        gradParams:zero()
        local predQ = model:forward(input)
        local maskLayer = nn.MaskedSelect()
        local predQOnActions = maskLayer:forward({predQ, actions})

        local loss = criterion:forward(predQOnActions, reward)
        local gradOutput = criterion(predQOnActions, reward)
        local gradMaskLayer = maskLayer:backward({predQ, actions}, gradOutput)
        model:backward(input,
            gradMaskLayer[1])
        return loss, gradParams    
    end
   
    local _, loss = optim.adam(feval, params, optimParams)
    print(epoch, rouge[streamSize + 1], loss[1])
    print(epsilon)
    print(qValues)
    print(actions)
    print(rouge)
    print(qValues:maskedSelect(actions))
    print(reward)

    --print(qValues:maskedSelect(actions))
    --print(reward)
    --print(actions)
    

    epsilon = epsilon / 1.001
end
