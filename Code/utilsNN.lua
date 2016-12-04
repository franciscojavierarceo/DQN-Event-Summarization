function scoreOracle(sentenceStream, maxSummarySize, refCounts)
    local SKIP = 1
    local SELECT = 2

    local buffer = Tensor(1, maxSummarySize):zero()
    local streamSize = sentenceStream:size(1)
    local actions = ByteTensor(streamSize, 2):fill(0)
    local summaryBuffer = LongTensor(streamSize + 1, maxSummarySize):zero()
    local oracleF1 = 0
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
        if f1 < oracleF1 then
            actions[i][SELECT] = 0
            actions[i][SKIP] = 1
        end
        if f1 > oracleF1 then
            oracleF1 = f1
        end
    end
    return oracleF1, actions
end

function buildModel(model, vocabSize, embeddingSize, metric, use_cuda)
    --- Small experiments seem to show that the Tanh activations performed better than the ReLU
    if model == 'bow' then
        print(string.format("Running bag-of-words model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    -- Note not averaging really blows up the model so keep this true
                    :add(nn.Sum(2, 3, true))
                    -- :add(nn.ReLU())
                    :add(nn.Tanh())
    else
        print(string.format("Running LSTM model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.SplitTable(2))
                    :add(nn.Sequencer(nn.LSTM(embeddingSize, embeddingSize)))
                    :add(nn.SelectTable(-1))            -- selects last state of the LSTM
                    :add(nn.Linear(embeddingSize, embeddingSize))
                    :add(nn.ReLU())
                    -- :add(nn.Tanh())
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
            :add(nn.Tanh())
            -- :add(nn.ReLU())
            :add(nn.Linear(embeddingSize * 3, 2))
            -- :add(nn.Tanh())
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

    if refTotal == 0 then 
        refTotal = 1 
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
    local actionMemory = torch.cat(newinput[3], memory_hist[3], 1)
    --- specifying rows to index 
    if sentMemory:size(1) >= memsize then
        -- My hack for sampling based on non-zero rewards
        local p = torch.abs(rewardMemory) / torch.abs(rewardMemory):sum()
        -- local p = torch.ones(memsize) / memsize
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

function stackMemory(newinput, memory_hist, memsize, use_cuda)
    local sentMemory = torch.cat(newinput[1][1]:double(), memory_hist[1][1]:double(), 1)
    local queryMemory = torch.cat(newinput[1][2]:double(), memory_hist[1][2]:double(), 1)
    local sumryMemory = torch.cat(newinput[1][3]:double(), memory_hist[1][3]:double(), 1)
    local rewardMemory = torch.cat(newinput[2]:double(), memory_hist[2]:double(), 1)
    local actionMemory = torch.cat(newinput[3], memory_hist[3], 1)
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

function backProp(input_memory, params, gradParams, optimParams, model, criterion, batch_size, n_backprops, use_cuda)
    local n = input_memory[1][1]:size(1)
    local p = torch.ones(n) / n
    local loss = 0
    for i=1, n_backprops do
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
            local lossf = criterion:forward(predQOnActions, reward)
            local gradOutput = criterion:backward(predQOnActions, reward)
            local gradMaskLayer = maskLayer:backward({predQ, actions_in}, gradOutput)
            model:backward(xinput, gradMaskLayer[1])
            return lossf, gradParams
        end
        --- optim.rmsprop returns \theta, f(\theta):= loss function
        _, lossv  = optim.rmsprop(feval, params, optimParams)   
        loss = loss + lossv[1]
    end
    return loss/n_backprops
end

function backPropOld(input_memory, params, gradParams, optimParams, model, criterion, batch_size, memsize, use_cuda)
    local inputs = {input_memory[1], input_memory[3]}
    local rewards = input_memory[2]
    local dataloader = dl.TensorLoader(inputs, rewards)
    for k, xin, reward in dataloader:sampleiter(batch_size, memsize) do
        xinput = xin[1]
        actions_in = xin[2]
        local function feval(params)
            gradParams:zero()
            local maskLayer = nn.MaskedSelect()
            if use_cuda then 
             maskLayer = maskLayer:cuda()
            end
            local predQ = model:forward(xinput)
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
function intialize_variables(query_file, inputs, n_samples, input_path, K_tokens, maxSummarySize)
    local vocabSize = 0
    local maxseqlen = 0
    local maxseqlenq = getMaxseq(query_file)
    local vocabSizeq = getVocabSize(query_file)
    local query_data = {}

    for query_id = 1, #inputs do
        input_fn = inputs[query_id]['inputs']
        nugget_fn = inputs[query_id]['nuggets']

        input_file = csvigo.load({path = input_path .. input_fn, mode = "large", verbose = false})
        nugget_file = csvigo.load({path = input_path .. nugget_fn, mode = "large", verbose = false})
        -- This is just for experimentation
        input_file = geti_n(input_file, 2,  n_samples) 
        -- input_file = geti_n(input_file, 2, #input_file) 
        nugget_file = geti_n(nugget_file, 2, #nugget_file) 

        vocabSize = math.max(vocabSize, vocabSizeq, getVocabSize(input_file))
        maxseqlen = math.max(maxseqlen, maxseqlenq, getMaxseq(input_file))
        xtdm  = buildTermDocumentTable(input_file, K_tokens)
        nuggets = buildTermDocumentTable(nugget_file, nil)
        ntdm = {}
        for i=1, #nuggets do
            ntdm = tableConcat(table.unpack(nuggets), ntdm)
        end
        -- Initializing the bookkeeping variables and storing them
        local query = LongTensor{inputs[query_id]['query'] }
        local sentenceStream = LongTensor(padZeros(xtdm, K_tokens))
        local streamSize = sentenceStream:size(1)
        local refSummary = Tensor{ntdm}
        local refCounts = buildTokenCounts(refSummary)
        local buffer = Tensor(1, maxSummarySize):zero()
        local actions = ByteTensor(streamSize, 2):fill(0)
        local actionsOpt = ByteTensor(streamSize, 2):fill(0)
        local exploreDraws = Tensor(streamSize)
        local summaryBuffer = LongTensor(streamSize + 1, maxSummarySize):zero()
        local qValues = Tensor(streamSize, 2):zero()
        local rouge = Tensor(streamSize + 1):zero()
        local rougeOpt = Tensor(streamSize + 1):zero()
        local summary = summaryBuffer:zero():narrow(1,1,1)
        local oracleF1, oracleActions = scoreOracle(sentenceStream, maxSummarySize, refCounts)

        query_data[query_id] = {
            sentenceStream,
            streamSize,
            query,
            actions,
            exploreDraws,
            summaryBuffer,
            qValues,
            rouge,
            actionsOpt,
            rougeOpt,
            refSummary,
            refCounts,
            buffer,
            oracleF1
        }
    end
    return vocabSize, query_data
end

function forwardpass(query_data, query_id, model, epsilon, gamma, metric, thresh, use_cuda)
    local SKIP = 1
    local SELECT = 2
    math.randomseed(420)
    torch.manualSeed(420)

    -- Extact variables
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
    local summary = summaryBuffer:zero():narrow(1,1,1)
    
    -- Have to set clear the inputs at the beginning of each scoring round
    actions:fill(0)
    actionsOpt:fill(0)
    rouge:fill(0)
    rougeOpt:fill(0)
    qValues:fill(0)
    summaryBuffer:fill(0)
    buffer:fill(0)
    exploreDraws:uniform(0, 1)

    for i=1, streamSize do     -- Iterating through individual sentences
        local sentence = sentenceStream:narrow(1, i, 1)
        qValues[i]:copy(model:forward({sentence, query, summary}))

        -- epsilon greedy strategy
        if exploreDraws[i]  <=  epsilon then        
            actions[i][torch.random(SKIP, SELECT)] = 1
        else 
            if qValues[i][SKIP] > qValues[i][SELECT] then
                actions[i][SKIP] = 1
            else
                actions[i][SELECT] = 1
            end
        end

        -- Storing the optimal predictions
        if qValues[i][SKIP] > qValues[i][SELECT] then
            actionsOpt[i][SKIP] = 1
        else
            actionsOpt[i][SELECT] = 1
        end
        local summary = buildSummary(
            actions:narrow(1, 1, i), 
            sentenceStream:narrow(1, 1, i),
            summaryBuffer:narrow(1, i + 1, 1),
            use_cuda
        )

        local summaryOpt = buildSummary(
            actionsOpt:narrow(1, 1, i), 
            sentenceStream:narrow(1, 1, i),
            summaryBuffer:narrow(1, i + 1, 1),
            use_cuda
        )

        local recall, prec, f1 = rougeScores(buildTokenCounts(summary), refCounts)
        local rOpt, pOpt, f1Opt = rougeScores(buildTokenCounts(summaryOpt), refCounts)

        if metric == "f1" then
            rouge[i + 1] = threshold(f1, thresh)
            rougeOpt[i]  = threshold(f1Opt, thresh)

        elseif metric == "recall" then
            rouge[i + 1] = threshold(recall, thresh)
            rougeOpt[i]  = threshold(rOpt, thresh)

        elseif metric == "precision" then
            rouge[i + 1] = threshold(prec, thresh)
            rougeOpt[i]  = threshold(pOpt, thresh)
        end

        if i==streamSize then
            rougeRecall = recall
            rougePrecision = prec
            rougeF1 = f1
        end
    end
    local max, argmax = torch.max(qValues, 2)
    local reward0 = rouge:narrow(1,2, streamSize) - rouge:narrow(1,1, streamSize)
    local reward_tm1 =  rougeOpt:narrow(1,2, streamSize) - rougeOpt:narrow(1,1, streamSize)
    local reward = reward0 + gamma * reward_tm1

    local querySize = query:size(2)
    local summaryBatch = summaryBuffer:narrow(1, 1, streamSize)
    local queryBatch = query:view(1, querySize):expand(streamSize, querySize) 
    local input = {sentenceStream, queryBatch, summaryBatch}
    local memory = {input, reward, actions}

    return memory, rougeRecall, rougePrecision, rougeF1, qValues
end

function train(inputs, query_data, model, nepochs, nnmod, metric, thresh, gamma, epsilon, delta, base_explore_rate, end_baserate, mem_size, batch_size, optimParams, n_backprops, use_cuda)
    math.randomseed(420)
    torch.manualSeed(420)
    criterion = nn.MSECriterion()
    local SKIP = 1
    local SELECT = 2

    if use_cuda then
        criterion = criterion:cuda()
        model = model:cuda()
    end

    local params, gradParams = model:getParameters()
    local perf = io.open(string.format("./Performance/Simulation/%s_%s_perf.txt", nnmod, metric), 'w')
    perf:write(string.format("epoch;epsilon;loss;randomF1;oracleF1;rougeF1;rougeRecall;rougePrecision;actual;pred;nselect;nskip;query\n"))
    for epoch=0, nepochs do
        for query_id=1, #inputs do
            -- Score the queries
            memory, rougeRecall, rougePrecision, rougeF1, qValues = forwardpass(
                            query_data, query_id, 
                            model, epsilon, gamma, 
                            metric, thresh, use_cuda
            )
            -- Build the memory
            if epoch == 0 then
                randomF1 = rougeF1
                if query_id == 1 and query_id ~= test_query then 
                    fullmemory = memory
                end
            else
                -- fullmemory = buildMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
                fullmemory = stackMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
            end
            --- Running backprop
            loss = backPropOld(fullmemory, params, gradParams, optimParams, model, criterion, batch_size, mem_size, use_cuda)

            nactions = torch.totable(memory[3]:sum(1))[1]
            perf_string = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}; %i; %i; %i\n", 
                epoch, epsilon, loss, randomF1, query_data[query_id][14],  -- this is the oracle
                rougeF1, rougeRecall, rougePrecision,
                memory[2]:min(), memory[2]:max(),
                qValues:min(), qValues:max(),
                nactions[SELECT], nactions[SKIP],
                query_id
            )
            perf:write(perf_string)

            local avpfile = io.open(string.format("./plotdata/%s/%i/%i_epoch.txt", nnmod, query_id, epoch), 'w')
            avpfile:write("predSkip;predSelect;actual;Skip;Select;query\n")
            for i=1, memory[1][1]:size(1) do
                avp_string = string.format("%.6f;%.6f;%6f;%i;%i;%i\n", 
                        qValues[i][SKIP], qValues[i][SELECT], memory[2][i], 
                        memory[3][i][SKIP], memory[3][i][SELECT], query_id)
                avpfile:write(avp_string)
            end
            avpfile:close()
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
    print(string.format("Model complete {Selected = %i; Skipped  = %i}; Final Rouge Recall, Precision, F1 = {%.6f;%.6f;%.6f}", 
                nactions[SELECT], nactions[SKIP], rougeRecall, rougePrecision, rougeF1))
end

function trainCV(inputs, query_data, model, nepochs, nnmod, metric, thresh, gamma, epsilon, delta, base_explore_rate, end_baserate, mem_size, batch_size, optimParams, n_backprops, use_cuda)
    math.randomseed(420)
    torch.manualSeed(420)
    criterion = nn.MSECriterion()
    local SKIP = 1
    local SELECT = 2

    if use_cuda then
        criterion = criterion:cuda()
        model = model:cuda()
    end

    local params, gradParams = model:getParameters()
    local perf = io.open(string.format("./Performance/CV/%s_%s_perf.txt", nnmod, metric), 'w')
    perf:write(string.format("epoch;epsilon;loss;randomF1;oracleF1;rougeF1;rougeRecall;rougePrecision;actual;pred;nselect;nskip;query\n"))

    for test_query=1, #inputs do
        for epoch=0, nepochs do
            for query_id=1, #inputs do
                -- Score the queries
                memory, rougeRecall, rougePrecision, rougeF1, qValues = forwardpass(
                                query_data, query_id, 
                                model, epsilon, gamma, 
                                metric, thresh, use_cuda
                )
                -- Build the memory
                if epoch == 0 then
                    randomF1 = rougeF1                    
                    if query_id == 1 and query_id ~= test_query then 
                        fullmemory = memory
                    end
                else
                    --- By not storing the memory of the test query we won't back prop on it
                    if query_id ~= test_query then 
                        -- fullmemory = buildMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
                        fullmemory = stackMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
                    end
                end
                -- Storing the test performance
                if query_id == test_query then
                    testf1 = rougeF1
                    if epoch==0 then
                        -- Only need to get these once
                        testrandomF1 = rougeF1
                        testOracleF1 = query_data[query_id][14]
                    end
                end
                --- Running backprop
                loss = backPropOld(fullmemory, params, gradParams, optimParams, model, criterion, batch_size, mem_size, use_cuda)

                nactions = torch.totable(memory[3]:sum(1))[1]
                perf_string = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}; %i; %i; %i\n", 
                    epoch, epsilon, loss, randomF1, query_data[query_id][14],  -- this is the oracle
                    rougeF1, rougeRecall, rougePrecision,
                    memory[2]:min(), memory[2]:max(),
                    qValues:min(), qValues:max(),
                    nactions[SELECT], nactions[SKIP],
                    query_id
                )
                perf:write(perf_string)
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
    end
    print(string.format("Model complete {Selected = %i; Skipped  = %i}; Final Rouge Recall, Precision, F1 = {%.6f;%.6f;%.6f}", 
                nactions[SELECT], nactions[SKIP], rougeRecall, rougePrecision, rougeF1))
end