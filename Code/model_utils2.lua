function buildModel(model, vocabSize, embeddingSize, use_cuda)
    if model == 'bow' then
        print(string.format("Running bag-of-words model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.Sum(2, 3, false))
                    :add(nn.ReLU())
                    -- :add(nn.Tanh())
    else
        print(string.format("Running LSTM model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.SplitTable(2))
                    :add(nn.Sequencer(nn.LSTM(embeddingSize, embeddingSize)))
                    :add(nn.SelectTable(-1))            -- selects last state of the LSTM
                    :add(nn.Linear(embeddingSize, embeddingSize))
                    :add(nn.ReLU())
    end
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
    if use_cuda then
        return nnmodel:cuda()
    end
    return nnmodel
end

function buildSummary(actions, sentences, buffer, use_cuda)
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
    if sentMemory:size(1) >= memsize then
        -- My hack for sampling uniformly distribution
        local p = torch.ones(memsize) / memsize
        local indxs = torch.multinomial(p, memsize, true)
        local sentMemory = sentMemory:index(1, indxs)
        local queryMemory = queryMemory:index(1, indxs)
        local sumryMemory = sumryMemory:index(1, indxs)
        local rewardMemory = rewardMemory:index(1, indxs)
        local actionMemory = actionMemory:index(1, indxs)
    end
    --- Selecting random samples of the data
    local inputMemory = {sentMemory, queryMemory, sumryMemory}
    if use_cuda then
        inputMemory = {sentMemory:cuda(), queryMemory:cuda(), sumryMemory:cuda()}
    end
    return {inputMemory, rewardMemory, actionMemory}
end

function backProp(input_memory, params, gradParams, optimParams, model, criterion, batch_size, use_cuda)
    local p = torch.ones(batch_size) / batch_size
    local indxs = torch.multinomial(p, batch_size, true)
    local xinput = {  
                input_memory[1][1]:index(1, indxs), 
                input_memory[1][2]:index(1, indxs), 
                input_memory[1][3]:index(1, indxs)
            }
    local reward = input_memory[2]:index(1, indxs)
    local actions_in = input_memory[3]:index(1, indxs)
    local function feval(params)
        gradParams:zero()
        local maskLayer = nn.MaskedSelect()
        if use_cuda then 
            maskLayer = maskLayer:cuda()
        end
        local predQ = model:forward(xinput)
        local predQOnActions = maskLayer:forward({predQ, actions_in})
        print(string.format("input1 = %i x %i", xinput[1]:size(1), xinput[1]:size(2)))
        print(string.format("input2 = %i x %i", xinput[2]:size(1), xinput[2]:size(2)))
        print(string.format("input3 = %i x %i", xinput[3]:size(1), xinput[3]:size(2)))
        print(string.format("actions= %i x %i", actions_in:size(1), actions_in:size(2)))
        print(string.format("rewards= %i x 1", reward:size(1)))
        print(string.format("predQ  = %i x %i", predQ:size(1), predQ:size(2)))
        print(string.format("predQOnActions = %i x 1", predQOnActions:size(1) ))
        print("****************************************")
        local lossf = criterion:forward(predQOnActions, reward)
        local gradOutput = criterion:backward(predQOnActions, reward)
        local gradMaskLayer = maskLayer:backward({predQ, actions_in}, gradOutput)
        model:backward(xinput, gradMaskLayer[1])
        return lossf, gradParams
    end
    --- optim.rmsprop returns \theta, f(\theta):= loss function
    _, lossv  = optim.rmsprop(feval, params, optimParams)   
    return lossv[1]
end