require 'os'
require 'nn'
require 'rnn'
require 'cunn'
require 'cunnx'
require 'optim'
require 'cutorch'
require 'parallel'

dl = require 'dataload'

-- Some useful functions
function genNbyK(n, k, a, b)
    torch.manualSeed(420)
    out = torch.LongTensor(n, k)
    for i=1, n do
        for j = 1, k do
            out[i][j] = torch.random(a, b)
        end
    end
    return out
end

function buildModel(model, vocabSize, embeddingSize, metric, adapt, use_cuda)
    -- Small experiments seem to show that the Tanh activations performed better\
    --      than the ReLU for the bow model
    if model == 'bow' then
        print(string.format("Running bag-of-words model to learn %s", metric))
        sentenceLookup = nn.Sequential()
                    :add(nn.LookupTableMaskZero(vocabSize, embeddingSize))
                    :add(nn.Sum(2, 3, true)) -- Not averaging blows up model so keep this true
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
    end
    local queryLookup = sentenceLookup:clone("weight", "gradWeight") 
    local summaryLookup = sentenceLookup:clone("weight", "gradWeight")
    local pmodule = nn.ParallelTable()
                :add(sentenceLookup)
                :add(queryLookup)
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
    --     actionmatrix = chosenactions:select(2, select_index):resize(1, n):view(n, 1):expand(n, k):clone()
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
            -- maxindex = torch.max(indices[torch.eq(totalPredsummary[i], 0)])
            -- totalPredsummary[i][{{maxindex - lenx + 1, maxindex}}]:copy(predsummary[i])
            -- Finding the smallest index with a zero
            minindex = torch.min(indices[torch.eq(tmpSummary[i], 0)])
            lenx = predsummary[i]:size(1)
            tmpSummary[i][{{minindex, minindex + lenx - 1}}]:copy(predsummary[i])
        end
    end
    return tmpSummary
end

function runSimulation(n, n_s, q, k, a, b, learning_rate, embDim, gamma, batch_size, fast, nepochs, epsilon, print_perf, mem_multiplier, cuts, base_explore_rate, endexplorerate, adapt, usecuda)
    -- torch.setnumthreads(16)
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

    optimParams = { learningRate = learning_rate }
    delta = cuts / nepochs
    end_baserate = torch.round(nepochs * endexplorerate )

    -- Simulating streams and queries
    queries = genNbyK(n, q, a, b)

    -- Note that the sentences are batched by sentence index so sentences[1] is the first sentence of each article
    sentences = {}
    for i=1, n_s do
        sentences[i] = genNbyK(n, k, a, b)
    end

    -- Optimal predicted summary
    trueSummary = LongTensor(n, k * n_s):fill(0)

    -- Using this to generate the optimal actions
    true_actions = {}
    for i=1, n_s do 
        ---- Simulating the data
        trueqValues = torch.rand(n, 2)
        
         ---- Generating the max values and getting the indices
        qMaxtrue, qindxtrue = torch.max(trueqValues, 2)
        
        --- I want to select the qindx elements for each row
        true_actions[i] = torch.zeros(n, 2):scatter(2, qindxtrue, torch.ones(trueqValues:size()))
        best_sentences = buildPredsummaryFast(true_actions[i], sentences[i], SELECT)
        trueSummary = buildTotalSummaryFast(best_sentences, trueSummary, usecuda)
    end

    qTokens = {}
    for i=1, n do
        qTokens[i] = Tokenize(trueSummary[i]:totable())
    end

    -- Building the model
    model = buildModel('lstm', b, embDim, 'f1', adapt, usecuda)
    params, gradParams = model:getParameters()
    if adapt then 
        criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.BCECriterion())
    else 
        criterion = nn.MSECriterion()
    end 

    totalPredsummary = {}
    qValues = {}
    qActions = {}
    qPreds = {}
    rewards = {}
    lossfull = {}
    rouguef1 = {}

    totalPredsummary = LongTensor(n, n_s * k):fill(0)

    memfull = false
    curr_memsize = 0
    memsize = n * n_s * mem_multiplier
    queryMemory = Tensor(memsize, q):fill(0)
    qActionMemory = Tensor(memsize, 2):fill(0)
    predSummaryMemory = Tensor(memsize, n_s * k):fill(0)
    sentenceMemory = Tensor(memsize, k):fill(0)
    qPredsMemory = Tensor(memsize, 2):fill(0)
    qValuesMemory = Tensor(memsize, 1):fill(0)
    rewardMemory = Tensor(memsize, 1):fill(0)

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
        if adapt then
            regPreds[i] = Tensor(n, 1):fill(0)
        end        
    end 

    if usecuda then
        criterion = criterion:cuda()
        model = model:cuda()
    end

    nClock = os.clock()
    for epoch=1, nepochs do
        --- Reset things at the start of each epoch
        for i=1, n_s do
            qPreds[i]:fill(0)
            qValues[i]:fill(0)
            qActions[i]:fill(0)
            rewards[i]:fill(0)
            totalPredsummary:fill(0)
            if adapt then
                regMemory[i]:fill(0)
            end        
        end

        for i=1, n_s do
            if torch.uniform(0, 1) <= epsilon then 
                qPreds[i]:copy(torch.rand(n, 2))
                -- Need to run a forward pass for the backward to work...wonky
                ignore = model:forward({sentences[i], queries, totalPredsummary})
            else 
                totalPreds = model:forward({sentences[i], queries, totalPredsummary})
                if adapt then 
                    qPreds[i]:copy(totalPreds[1])
                    regPreds[i]:copy(totalPreds[2])
                else
                    qPreds[i]:copy(totalPreds)
                end
            end 

            if fast then 
                qMax, qindx = torch.max(qPreds[i], 2)  -- Pulling the best actions
                -- Here's the fast way to select the optimal action for each query
                qActions[i]:copy(qActions[i]:scatter(2, qindx, torch.ones(qPreds[i]:size())):clone())
                qValues[i]:copy(qMax)
                predsummary = buildPredsummaryFast(qActions[i], sentences[i], SELECT)
                totalPredsummary = buildTotalSummaryFast(predsummary, totalPredsummary, usecuda)
            else 
                for j=1, n do
                    if qPreds[i][j][SELECT] > qPreds[i][j][SKIP] then
                        qActions[i][j][SELECT] = 1
                        qValues[i][j]:fill(qPreds[i][j][SELECT])
                    else
                        qActions[i][j][SKIP] = 1
                        qValues[i][j]:fill(qPreds[i][j][SKIP])
                    end
                end
                predsummary = buildPredsummary(predsummary, qActions[i], sentences[i], SELECT)
                buildTotalSummary(predsummary, totalPredsummary)
            end
            for j = 1, n do
                recall, prec, f1 = rougeScores( qTokens[j],
                                                Tokenize(totalPredsummary[j]:totable()))
                rewards[i][j]:fill(f1)
            end
            if i > 1 then
                -- Calculating change in rougue f1
                rewards[i]:copy(rewards[i] - rewards[i-1])
            end
            -- here's the row indexing
            start_row = curr_memsize + 1
            if memsize < (start_row + n) then 
                start_row = memsize - n + 1
                end_row = start_row + n - 1
                memfull = true
                curr_memsize = 0
            else 
                end_row = start_row + n - 1
                curr_memsize = end_row
            end            
            -- Update memory sequentially until it's full then restart updating it
            qActionMemory[{{start_row, end_row}}]:copy(qActions[i])
            predSummaryMemory[{{start_row, end_row}}]:copy(totalPredsummary)
            sentenceMemory[{{start_row, end_row}}]:copy(sentences[i])
            qPredsMemory[{{start_row, end_row}}]:copy(qPreds[i])
            qValuesMemory[{{start_row, end_row}}]:copy(qValues[i])
            queryMemory[{{start_row, end_row}}]:copy(queries)
            if adapt then
                regMemory[{{start_row, end_row}}]:copy(regPreds[i])
            end        
        end
        for i=1, n_s do
            if i  < n_s then
                rewardMemory[{{n * (i-1) + 1, n * i}}]:copy(rewards[i] + gamma * rewards[i + 1] )
            else
                rewardMemory[{{n * (i-1) + 1, n * i}}]:copy(rewards[i] )
            end
        end
        -- Adding back the delta for the last one
        rouguef1[epoch] = (rewards[n_s] + rewards[ n_s - 1] ):mean()

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
                            qValuesMemory[{{1, memrows}}]:cuda()
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
                        qValuesMemory[{{1, memrows}}]
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
                    local ignore = model:forward({xin[1], xin[2], xin[3]})
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
                    local ignore = model:forward({xin[1], xin[2], xin[3]})
                    local predQOnActions = maskLayer:forward({xin[4], xin[5]}) 
                    lossf = criterion:forward(predQOnActions, reward)
                    local gradOutput = criterion:backward(predQOnActions, reward)
                    local gradMaskLayer = maskLayer:backward({xin[4], xin[5]}, gradOutput)
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

cmd = torch.CmdLine()
cmd:option('--fast', true, 'implement fast indexing')
cmd:option('--n_samples', 100, 'Number of queries')
-- n_samples = 100000 will reproduce the speed numbers
cmd:option('--n_s', 5, 'Number of sentences')
cmd:option('--q_l', 5, 'Query length')
cmd:option('--k', 7, 'Number of samples to iterate over')
cmd:option('--a', 1, 'Number of samples to iterate over')
cmd:option('--b', 100, 'Number of samples to iterate over')
cmd:option('--lr', 0.000001, 'Learning rate')
cmd:option('--embDim', 100, 'Number of samples to iterate over')
cmd:option('--gamma', 0., 'Weight of future prediction')
cmd:option('--batch_size', 25, 'Batch size')
cmd:option('--memory_multiplier', 1, 'Multiplier defining size of memory')
cmd:option('--cuts', 4, 'How long we want to decay search over')
cmd:option('--endexplorerate', 0.8, 'When to end the exploration as a percent of trainings epochs)')
cmd:option('--base_explore_rate', 0.1, 'Base exploration rate after 1/cuts until endexplorerate')
cmd:option('--nepochs', 100, 'Number of epochs')
cmd:option('--epsilon', 1, 'Random sampling rate')
cmd:option('--print', false, 'print performance')
cmd:option('--adapt', false, 'Use adaptive regularization')
cmd:option('--usecuda', false, 'cuda option')
cmd:text()
local opt = cmd:parse(arg or {})       --- stores the commands in opt.variable (e.g., opt.model)

-- Running the script
runSimulation(opt.n_samples, opt.n_s, opt.q_l, opt.k, opt.a, opt.b, opt.lr,
              opt.embDim, opt.gamma, opt.batch_size, opt.fast, opt.nepochs, opt.epsilon, opt.print, 
              opt.memory_multiplier, opt.cuts, opt.base_explore_rate, opt.endexplorerate, 
              opt.adapt, opt.usecuda)

-- Notes
-- 2. Optimize using masklayer
-- 7. Deploy the model in chunks?
