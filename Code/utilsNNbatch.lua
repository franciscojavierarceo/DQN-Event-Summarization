require 'os'
require 'nn'
require 'rnn'
-- require 'cunn'
-- require 'cunnx'
require 'optim'
-- require 'cutorch'
require 'parallel'

dl = require 'dataload'

-- Some useful functions
function genNbyK(n, k, a, b)
    out = torch.LongTensor(n, k)
    for i=1, n do
        for j = 1, k do
            out[i][j] = torch.random(a, b)
        end
    end
    return out
end

-- local lu = nn.LookupTableMaskZero(vocabSize, embeddingSize)
-- local sentenceEncoder = nn.Sequential()
--     :add(lu)
--     :add(nn.Sum(2, 3, true))
--     :add(nn.Linear(embeddingSize, embeddingSize))
--     :add(nn.Tanh())
-- local queryEncoder = nn.Sequential()
--     :add(lu:clone("weight", "gradWeight"))
--     :add(nn.Sum(2, 3, true))
--     :add(nn.Linear(embeddingSize, embeddingSize))
--     :add(nn.Tanh())
-- local summaryEncoder = nn.Sequential()
--     :add(lu:clone("weight", "gradWeight"))
--     :add(nn.Sum(2, 3, true))
--     :add(nn.Linear(embeddingSize, embeddingSize))
--     :add(nn.Tanh())

-- local pmodule = nn.ParallelTable()
--     :add(queryEncoder)
--     :add(sentenceEncoder)
--     :add(summaryEncoder)
-- local nnmodel = nn.Sequential()
--     :add(pmodule)
--     :add(nn.JoinTable(2))
--     :add(nn.Linear(embeddingSize * 3, embeddingSize))
--     :add(nn.Tanh())
--     :add(nn.Linear(embeddingSize, 2))
-- return nnmodel

function buildModel(model, vocabSize, embeddingSize, metric, adapt, use_cuda)
    -- Small experiments seem to show that the Tanh activations performed better\
    --      than the ReLU for the bow model
    if model == 'bow' then
        print(string.format("Running bag-of-words model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    -- Not averaging blows up model so keep this true
                    :add(nn.Sum(2, 3, true)) 
                    :add(nn.Tanh())
    else
    -- This needs to have a transpose in the model
    -- lstms go 
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
                :add(queryLookup)
                :add(sentenceLookup)
                :add(summaryLookup)

    if model == 'bow' then
        nnmodel = nn.Sequential()
            :add(pmodule)
            :add(nn.JoinTable(2))
            :add(nn.Tanh())
            :add(nn.Linear(embeddingSize * 3, 2))
    else
        nnmodel = nn.Sequential()
            :add(pmodule)
            :add(nn.JoinTable(2))
            :add(nn.ReLU())
            :add(nn.Linear(embeddingSize * 3, 2))
    end

    if adapt then 
        print("Adaptive regularization")
        local logmod = nn.Sequential()
            :add(nn.Linear(embeddingSize * 3, 1))
            :add(nn.LogSigmoid())
            :add(nn.SoftMax())

        local regmod = nn.Sequential()
            :add(nn.Linear(embeddingSize * 3, 2))

        local fullmod = nn.ConcatTable()
            :add(regmod)
            :add(logmod)

        local final = nn.Sequential()
            :add(pmodule)
            :add(nn.JoinTable(2))
            :add(fullmod)

        nnmodel = final
    end

    if use_cuda then
        return nnmodel:cuda()
    end
    return nnmodel
end

function Tokenize(inputdic)
    --- This function tokenizes the words into a unigram dictionary
    local out = {}
    for k, v in pairs(inputdic) do
        if v ~= 0 then 
            if out[v] == nil then
                out[v] = 1
            else 
                out[v] = 1 + out[v]
            end
        end
    end
    return out
end

function rougeScores(genSummary, refSummary)
    local genTotal = 0
    local refTotal = 0
    local intersection = 0
    -- Inserting the missing keys
    for k, genCount in pairs(genSummary) do
        if refSummary[k] == nil then
            refSummary[k] = 0
        end
    end
    for k, refCount in pairs(refSummary) do
        local genCount = genSummary[k]
        if genCount == nil then 
            genCount = 0 
        end
        intersection = intersection + math.min(refCount, genCount)
        refTotal = refTotal + refCount
        genTotal = genTotal + genCount
    end

    recall = intersection / refTotal
    prec = intersection / genTotal
    if refTotal == 0 then
        recall = 0
    end 
    if genTotal == 0 then
        prec = 0
    end
    -- tmp = {intersection, refTotal, genTotal}
    if recall > 0 or prec > 0 then
        f1 = (2 * recall * prec) / (recall + prec)
    else 
        f1 = 0
    end
    return recall, prec, f1
end

function buildPredsummary(summary, chosenactions, inputsentences, select_index)
    if summary == nil then
        summary = torch.zeros(inputsentences:size())
    end
    for i=1, chosenactions:size(1) do
        -- the 2 is for the SELECT index, will have to make this more general later
        if chosenactions[i][select_index] == 1 then
            summary[i]:copy(inputsentences[i])
        end
    end    
    return summary
end

function buildPredsummaryFast(chosenactions, inputsentences, select_index)
    local n = inputsentences:size(1)
    local k = inputsentences:size(2)
    local summary = torch.zeros(inputsentences:size())
    actionmatrix = chosenactions:select(2, select_index):clone():resize(n, 1):view(n, 1):expand(n, k):clone()
    --     This line didn't work for whatever reason...gives weird indexing...
    return actionmatrix:cmul(inputsentences:double())
end

function buildTotalSummary(predsummary, totalPredsummary)
    nps = predsummary:size(1)
    n_l = totalPredsummary:size(2)
    indices = torch.linspace(1, n_l, n_l):long() 
    for i=1, predsummary:size(1) do
        if predsummary[i]:sum() > 0 then 
            minindex = 1
            for j = 1, totalPredsummary[i]:size(1) do 
                if totalPredsummary[i][j] > 0 then
                    minindex = minindex + 1
                end
            end
            lenx = predsummary[i]:size(1)
            totalPredsummary[i][{{minindex, minindex + lenx - 1}}]:copy(predsummary[i])
        end
    end
end

function buildTotalSummaryFast(predsummary, inputTotalSummary, usecuda)
    tmpSummary = inputTotalSummary:clone()
    nps = predsummary:size(1)
    n_l = inputTotalSummary:size(2)    
    indices = torch.linspace(1, n_l, n_l):long()
    if usecuda then
        indices = indices:cuda()
    end
    for i=1, predsummary:size(1) do
        if predsummary[i]:sum() > 0 then
            -- Finding the largest index with a zero
            -- maxindex = torch.max(indices[torch.eq(tmpSummary[i], 0)])
            -- lenx = predsummary[i]:size(1)
            -- tmpSummary[i][{{maxindex - lenx + 1, maxindex}}]:copy(predsummary[i])
            -- Finding the smallest index with a zero
            minindex = torch.min(indices[torch.eq(tmpSummary[i], 0)])
            lenx = predsummary[i]:size(1)
            tmpSummary[i][{{minindex, minindex + lenx - 1}}]:copy(predsummary[i])
        end
    end
    return tmpSummary
end

function train(queries, sentences, trueSummaries, learning_rate, vocab_size, embDim, gamma, batch_size, nepochs, epsilon, print_perf, mem_multiplier, cuts, base_explore_rate, endexplorerate, adapt, adapt_lambda, usecuda, seedval)
    -- torch.setnumthreads(16)
    dofile("Code/Utils/load_cnn.lua")
    torch.manualSeed(seedval)
    if usecuda then
        Tensor = torch.CudaTensor
        LongTensor = torch.CudaLongTensor   
        ByteTensor = torch.CudaByteTensor
        maskLayer = nn.MaskedSelect():cuda()
        print("...running on GPU")
    else
        Tensor = torch.Tensor
        LongTensor = torch.LongTensor
        ByteTensor = torch.ByteTensor
        maskLayer = nn.MaskedSelect()
        print("...running on CPU")
    end

    local SKIP = 1
    local SELECT = 2

    k = sentences[1]:size(2)
    n = queries:size(1)
    q = queries:size(2)
    n_s = #sentences

    optimParams = { learningRate = learning_rate }
    delta = cuts / nepochs
    end_baserate = torch.round(nepochs * endexplorerate )


    qTokens = {}
    for i=1, n do
        qTokens[i] = Tokenize({trueSummaries[i]:totable()}, false)[1]
    end

    -- Building the model
    model = buildModel('bow', vocab_size, embDim, 'f1', adapt, usecuda)
    params, gradParams = model:getParameters()

    if adapt then 
        criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.BCECriterion())
        criterion["weights"] = {1, adapt_lambda}
    else 
        criterion = nn.MSECriterion()
    end 

    print(string.format("Running model with %i queries and %i sentences", n, n_s)) 

    qValues = {}
    qActions = {}
    qPreds = {}
    rewards = {}
    lossfull = {}
    rouguef1 = {}
    rougue_scores = {}

    totalPredsummary = LongTensor(n, n_s * k):fill(0)

    memfull = false
    curr_memsize = 0
    memsize = n * n_s * mem_multiplier
    queryMemory = Tensor(memsize, q):fill(0)
    qActionMemory = Tensor(memsize, 2):fill(0)
    predSummaryMemory = Tensor(memsize, n_s * k):fill(0)
    sentenceMemory = Tensor(memsize, k):fill(0)
    sentencetp1Memory  = Tensor(memsize, k):fill(0)
    predSummarytp1Memory = Tensor(memsize, n_s * k):fill(0)
    qPredsMemory = Tensor(memsize, 2):fill(0)
    qValuesMemory = Tensor(memsize, 1):fill(0)
    rewardMemory = Tensor(memsize, 1):fill(0)
    totalPreds = Tensor(n, 2):fill(0)

    if adapt then
        regPreds = {}
        regMemory = Tensor(memsize, 1):fill(0) 
    end
    --- Initializing thingss
    for i = 1, n_s do
        qPreds[i] = Tensor(n, 2):fill(0) 
        qValues[i] = Tensor(n, 1):fill(0)
        qActions[i] = Tensor(n, 2):fill(0)
        rewards[i] = Tensor(n, 1):fill(0)
        rougue_scores[i] = Tensor(n, 1):fill(0)
        if adapt then
            regPreds[i] = Tensor(n, 1):fill(0)
        end        
    end 

    if usecuda then
        criterion = criterion:cuda()
        model = model:cuda()
    end
    print("training...")

    nClock = os.clock()
    for epoch=1, nepochs do
        --- Reset things at the start of each epoch
        for i=1, n_s do
            qPreds[i]:fill(0)
            qValues[i]:fill(0)
            qActions[i]:fill(0)
            rewards[i]:fill(0)
            rougue_scores[i]:fill(0)
            totalPredsummary:fill(0)
            if adapt then
                regMemory[i]:fill(0)
            end        
        end

        for i=1, n_s do
            totalPreds:fill(0)
            start_row = 1
            end_row = batch_size
            c = 1             
            for start_row=1, n, batch_size do                 
                end_row = c * batch_size

                if end_row > n then 
                    end_row = n
                end
                totalPreds[{{start_row, end_row}}]:copy(
                    model:forward({
                        queries[{{start_row, end_row}}], 
                        sentences[i][{{start_row, end_row}}], 
                        totalPredsummary[{{start_row, end_row}}]
                    })
                )
                c = c + 1
            end

            if adapt then 
                qPreds[i]:copy(totalPreds[1])
                regPreds[i]:copy(totalPreds[2])
            else
                qPreds[i]:copy(totalPreds)
            end

            if torch.uniform(0, 1) <= epsilon then
                -- randomly choosing actions
                xrand = torch.rand(qPreds[i]:size())
                qActions[i]:select(2, SELECT):copy(torch.ge(xrand:select(2, SELECT), xrand:select(2, SKIP)))
                qActions[i]:select(2, SKIP):copy(torch.ge(xrand:select(2, SKIP), xrand:select(2, SELECT)))
                qValues[i]:copy( maskLayer:forward({totalPreds, qActions[i]:byte()}) )
            else 
                qMax, qindx = torch.max(qPreds[i], 2)  -- Pulling the best actions
                -- Here's the fast way to select the optimal action for each query
                qActions[i]:copy(
                    qActions[i]:scatter(2, qindx, torch.ones(qPreds[i]:size())):clone()
                )
                qValues[i]:copy(
                    qMax
                )
            end

            -- This is where we begin to store the data in our memory 
                -- notice that we store the reward after this part
            start_row = curr_memsize + 1
            if memsize < (start_row + n) then 
                start_row = memsize - n + 1
                end_row = start_row + n - 1
                curr_memsize = 0
                if (end_row + n) >= memsize then 
                    memfull = true
                end 
            else 
                end_row = start_row + n - 1
                curr_memsize = end_row
            end

            -- Update memory sequentially until it's full then restart updating it
            queryMemory[{{start_row, end_row}}]:copy(queries)
            sentenceMemory[{{start_row, end_row}}]:copy(sentences[i])
            predSummaryMemory[{{start_row, end_row}}]:copy(totalPredsummary)
            
            -- Now that we've stored our memory, we can build the summary to evaluate our action
            predsummary = buildPredsummaryFast(qActions[i], sentences[i], SELECT)
            totalPredsummary = buildTotalSummaryFast(predsummary, totalPredsummary, usecuda)
            
            if i < n_s then
                sentencetp1Memory[{{start_row, end_row}}]:copy(sentences[i + 1])
                predSummarytp1Memory[{{start_row, end_row}}]:copy(totalPredsummary)
            else 
                sentencetp1Memory[{{start_row, end_row}}]:copy(Tensor(sentences[i]:size()):fill(0) )
                predSummarytp1Memory[{{start_row, end_row}}]:copy(Tensor(totalPredsummary:size()):fill(0) )
            end 
            
            qActionMemory[{{start_row, end_row}}]:copy(qActions[i])
            qPredsMemory[{{start_row, end_row}}]:copy(qPreds[i])
            qValuesMemory[{{start_row, end_row}}]:copy(qValues[i])

            if adapt then
                regMemory[{{start_row, end_row}}]:copy(regPreds[i])
            end

            for j = 1, n do
                recall, prec, f1 = rougeScores( Tokenize(totalPredsummary[j]:totable()),
                                                qTokens[j]
                    )
                rougue_scores[i][j]:fill(f1)
            end

            if i == n_s then 
                rouguef1[epoch] = rougue_scores[i]:mean()
            end 

            if i == 1 then
                -- Calculating change in rougue f1
                rewards[i]:copy(rougue_scores[i])
            else 
                rewards[i]:copy(rougue_scores[i] - rougue_scores[i-1])
            end
            rewardMemory[{{start_row, end_row}}]:copy(rewards[i])
        end

        if memfull then 
            memrows = memsize
        else 
            memrows = curr_memsize
        end

        if usecuda then 
            dataloader = dl.TensorLoader({
                            queryMemory[{{1, memrows}}]:cuda(), 
                            sentenceMemory[{{1, memrows}}]:cuda(), 
                            predSummaryMemory[{{1, memrows}}]:cuda(),
                            qPredsMemory[{{1, memrows}}]:cuda(), 
                            ByteTensor(memrows, 2):copy(qActionMemory[{{1, memrows}}]), 
                            qValuesMemory[{{1, memrows}}]:cuda(),
                            sentencetp1Memory[{{1, memrows}}]:cuda(),
                            predSummarytp1Memory[{{1, memrows}}]:cuda()               
                                }, 
                        rewardMemory[{{1, memrows}}]:cuda()
                    )
            if adapt then            
                table.insert(dataloader['inputs'], regMemory[{{1, memrows}}]:cuda() )
            end
        else 
            dataloader = dl.TensorLoader({
                        queryMemory[{{1, memrows}}], 
                        sentenceMemory[{{1, memrows}}], 
                        predSummaryMemory[{{1, memrows}}], 
                        qPredsMemory[{{1, memrows}}], 
                        ByteTensor(memrows, 2):copy(qActionMemory[{{1, memrows}}]), 
                        qValuesMemory[{{1, memrows}}],
                        sentencetp1Memory[{{1, memrows}}],
                        predSummarytp1Memory[{{1, memrows}}]                    
                        }, 
                    rewardMemory[{{1, memrows}}]
                )
            if adapt then
                table.insert(dataloader['inputs'], regMemory[{{1, memrows}}] )
            end
        end
        loss = {}
        c = 1
        for k, xin, reward in dataloader:sampleiter(batch_size, memsize) do
            local function feval(params)
                gradParams:zero()
                if adapt then
                    local predtp1 = model:forward({xin[1], xin[7], xin[8]})
                    local predQOnActions = maskLayer:forward({xin[4], xin[5]}) 
                    local ones = torch.ones(reward:size(1)):resize(reward:size(1))
                    if usecuda then
                        ones = ones:cuda()
                    end
                    lossf = criterion:forward({predQOnActions, xin[7]}, {reward, ones})
                    local gradOutput = criterion:backward({predQOnActions, xin[6]}, {reward, ones})
                    local gradMaskLayer = maskLayer:backward({xin[4], xin[5]}, gradOutput[1])
                    model:backward({xin[1], xin[2], xin[3]}, {gradMaskLayer[1], gradOutput[2]})
                else
                    model:forget()
                    local predtp1 = model:forward({xin[1], xin[7], xin[8]})
                    local predtp1max, _ = torch.max(predtp1, 2)
                    model:forget()
                    local predt = model:forward({xin[1], xin[2], xin[3]})
                    local y_j = reward + (gamma * predtp1max) 
                    local predQOnActions = maskLayer:forward({predt, xin[5]}) 
                    lossf = criterion:forward(predQOnActions, y_j )
                    local gradOutput = criterion:backward(predQOnActions, y_j)
                    local gradMaskLayer = maskLayer:backward({predt, xin[5]}, gradOutput)
                    model:backward({xin[1], xin[2], xin[3]}, gradMaskLayer[1])

                end 
                return lossf, gradParams
            end
            --- optim.rmsprop returns \theta, f(\theta):= loss function
             _, lossv  = optim.rmsprop(feval, params, optimParams)
            loss[c] = lossv[1]
            c = c + 1
        end

        lossfull[epoch] = Tensor(loss):sum() / #lossv
        if print_perf then
            print(
                string.format('epoch = %i; rougue = %.6f; epsilon = %.6f; loss = %.6f' , 
                    epoch, rouguef1[epoch], epsilon, lossfull[epoch])
                )
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
    print(string.format("Elapsed time: %.5f" % (os.clock()-nClock) ))
    print(
        string.format('First rougue = %.6f; Last rougue = %.6f',
            rouguef1[1], rouguef1[nepochs]) 
        )
end

-- th Code/Simulations/DQN_Batch_Queries_Simulation.lua --n_samples 100 --lr 1e-2 --n_s 10  --k 10 --q_l 4 --a 1 --b 1000 --gamma 0.4 --print --base_explore_rate 0.1 --endexplorerate 0.5 --nepochs 200 --cuts 4 --memory_multiplier 3 --batch_size 25 --embDim 50 --seedval 100
