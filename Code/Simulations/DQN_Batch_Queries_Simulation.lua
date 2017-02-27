require 'os'
require 'nn'
require 'rnn'
require 'optim'
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
        if out[v] == nil then
            out[v] = 1
        else 
            out[v] = 1 + out[v]
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

function buildPredsummaryFast(summary, chosenactions, inputsentences, select_index)
    n = inputsentences:size(1)
    k = inputsentences:size(2)
    if summary == nil then
        summary = torch.zeros(inputsentences:size())
    end
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
            -- maxindex = 0
            -- for j = 1, totalPredsummary[i]:size(1) do 
            --     if totalPredsummary[i][j] == 0 then
            --         maxindex = maxindex + 1
            --     end
            -- end
            -- lenx = predsummary[i]:size(1)
            -- totalPredsummary[i][{{maxindex - lenx + 1, maxindex}}]:copy(predsummary[i])

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

function buildTotalSummaryFast(predsummary, totalPredsummary)
    nps = predsummary:size(1)
    n_l = totalPredsummary:size(2)
    indices = torch.linspace(1, n_l, n_l):long() 
    for i=1, predsummary:size(1) do
        if predsummary[i]:sum() > 0 then 
            -- Finding the largest index with a zero
            -- maxindex = torch.max(indices[torch.eq(totalPredsummary[i], 0)])
            -- totalPredsummary[i][{{maxindex - lenx + 1, maxindex}}]:copy(predsummary[i])
            -- Finding the smallest index with a zero
            minindex = torch.min(indices[torch.eq(totalPredsummary[i], 0)])
            lenx = predsummary[i]:size(1)
            totalPredsummary[i][{{minindex, minindex + lenx - 1}}]:copy(predsummary[i])
        end
    end
end

function runSimulation(n, n_s, q, k, a, b, embDim, fast, nepochs, epsilon, print_perf)
    local SKIP = 1
    local SELECT = 2
    batch_size = 10
    gamma = 0
    maskLayer = nn.MaskedSelect()
    optimParams = { learningRate = 0.0001 }

    -- Simulating streams and queries
    queries = genNbyK(n, q, a, b)

    -- Note that the sentences are batched by sentence index so sentences[1] is the first sentence of each article
    sentences = {}
    for i=1, n_s do
        sentences[i] = genNbyK(n, k, a, b)
    end

    -- Optimal predicted summary
    trueSummary = torch.zeros(n, k * n_s)
    -- Using this to generate the optimal actions
    true_actions = {}
    for i=1, n_s do 
        ---- Simulating the data
        trueqValues = torch.rand(n, 2)
        
         ---- Generating the max values and getting the indices
        qMaxtrue, qindxtrue = torch.max(trueqValues, 2)
        
        --- I want to select the qindx elements for each row
        true_actions[i] = torch.zeros(n, 2):scatter(2, qindxtrue, torch.ones(trueqValues:size()))
        best_sentences = buildPredsummaryFast(best_sentences, true_actions[i], sentences[i], SELECT)
        buildTotalSummaryFast(best_sentences, trueSummary)
    end

    qTokens = {}
    for i=1, n do
        qTokens[i] = Tokenize(trueSummary[i]:totable())
    end

    -- Building the model
    model = buildModel('bow', b, embDim, 'f1', false, false)
    params, gradParams = model:getParameters()
    criterion = nn.MSECriterion()

    totalPredsummary = {}
    qValues = {}
    qActions = {}
    qPreds = {}
    rewards = {}
    lossfull = {}
    rouguef1 = {}

    memsize = n * n_s
    queryMemory = torch.zeros(memsize, q)
    qActionMemory = torch.zeros(memsize, 2)
    predSummaryMemory = torch.zeros(memsize, n_s * k)
    sentenceMemory = torch.zeros(memsize, k)
    qPredsMemory = torch.zeros(memsize, 2)
    qValuesMemory = torch.zeros(memsize, 1)
    rewardMemory = torch.zeros(memsize, 1)

    nClock = os.clock()
    for epoch=1, nepochs do
        for i = 1, n_s do
            --- Initializing things
            if epoch == 1 then 
                qPreds[i] = torch.zeros(n, 2)
                qValues[i] = torch.zeros(n, 1) 
                qActions[i] = torch.zeros(n, 2)
                rewards[i] = torch.zeros(n, 1)
                totalPredsummary[i] = torch.LongTensor(n, n_s * k):fill(0)
            else
                --- Reset things
                qPreds[i]:fill(0)
                qValues[i]:fill(0)
                qActions[i]:fill(0)
                rewards[i]:fill(0)
                totalPredsummary[i]:fill(0)
            end 
        end
        for i=1, n_s do
            if torch.uniform(0, 1) <= epsilon then 
                qPreds[i]:copy(torch.rand(n, 2))
                -- Need to run a forward pass for the backward to work...wonky
                ignore = model:forward({sentences[i], queries, totalPredsummary[i]})
            else 
                qPreds[i]:copy(model:forward({sentences[i], queries, totalPredsummary[i]}) )
            end 
            if fast then 
                qMax, qindx = torch.max(qPreds[i], 2)  -- Pulling the best actions
                -- Here's the fast way to select the optimal action for each query
                -- qActions[i] = torch.zeros(n, 2):scatter(2, qindx, torch.ones(qPreds[i]:size())):clone()
                qActions[i]:copy(qActions[i]:scatter(2, qindx, torch.ones(qPreds[i]:size())):clone())
                qValues[i]:copy(qMax)
                predsummary = buildPredsummaryFast(predsummary, qActions[i], sentences[i], SELECT)
                buildTotalSummaryFast(predsummary, totalPredsummary[i])
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
                buildTotalSummary(predsummary, totalPredsummary[i])
            end
            for j = 1, n do
                recall, prec, f1 = rougeScores( qTokens[j],
                                                Tokenize(totalPredsummary[i][j]:totable()))
                rewards[i][j]:fill(f1)
            end
            if i > 1 then
                -- Calculating change in rougue f1
                rewards[i]:copy(rewards[i] - rewards[i-1])
            end
            -- Update memory sequentially until it's full 
            qActionMemory[{{n * (i-1) + 1, n * i}}]:copy(qActions[i])
            predSummaryMemory[{{n * (i-1) + 1, n * i}}]:copy(totalPredsummary[i])
            sentenceMemory[{{n * (i-1) + 1, n * i}}]:copy(sentences[i])
            qPredsMemory[{{n * (i-1) + 1, n * i}}]:copy(qPreds[i])
            qValuesMemory[{{n * (i-1) + 1, n * i}}]:copy(qValues[i])
            queryMemory[{{n * (i-1) + 1, n * i}}]:copy(queries)
            if i  < n_s then
                rewardMemory[{{n * (i-1) + 1, n * i}}]:copy(rewards[i] + gamma * rewards[i + 1] )
            else
                rewardMemory[{{n * (i-1) + 1, n * i}}]:copy(rewards[i] )
            end
        end
        -- Adding back the delta for the last one
        rouguef1[epoch] = (rewards[n_s] + rewards[ n_s - 1] ):mean()
        -- rouguef1[epoch] = rewards[n_s]:mean()

        loss = {}
        local dataloader = dl.TensorLoader({queryMemory, sentenceMemory, predSummaryMemory, qPredsMemory, qActionMemory, qValuesMemory}, rewardMemory)
        c = 1
        for k, xin, reward in dataloader:sampleiter(batch_size, memsize) do
            local function feval(params)
                gradParams:zero()
                if adapt then
                    local ignore = model:forward({xin[1], xin[2], xin[3]})
                    local predQOnActions = maskLayer:forward({qPredsMemory, actions_in}) 
                    local ones = torch.ones(predQ:size(1)):resize(predQ:size(1))
                    lossf = criterion:forward({qValuesMemory, predReg}, {reward, ones})
                    local gradOutput = criterion:backward({qActionMemory, predReg}, {reward, ones})
                    local gradMaskLayer = maskLayer:backward({qPredsMemory, qActionMemory}, gradOutput[1])
                    model:backward({queryMemory, sentenceMemory, predSummaryMemory}, {gradMaskLayer[1], gradOutput[2]})
                else 
                    local ignore = model:forward({xin[1], xin[2], xin[3]})
                    local predQOnActions = maskLayer:forward({xin[4], xin[5]:byte()}) 
                    lossf = criterion:forward(predQOnActions, reward)
                    local gradOutput = criterion:backward(predQOnActions, reward)
                    local gradMaskLayer = maskLayer:backward({xin[4], xin[5]:byte()}, gradOutput)
                    model:backward({xin[1], xin[2], xin[3]}, gradMaskLayer[1])
                end 
                return lossf, gradParams
            end
            --- optim.rmsprop returns \theta, f(\theta):= loss function
             _, lossv  = optim.rmsprop(feval, params, optimParams)
            loss[c] = lossv[1]
            c = c + 1
        end

        lossfull[epoch] = torch.Tensor(loss):sum() / #lossv
        if print_perf then
            print(
                string.format('epoch = %i; rougue = %.6f; epsilon = %.6f; loss = %.6f' , 
                    epoch, rouguef1[epoch], epsilon, lossfull[epoch]) 
                )
        end
        epsilon = epsilon - (1/10.)
        if epsilon < 0 then
            epsilon = 0
        end
    end
    print(string.format("Elapsed time: %.5f" % (os.clock()-nClock) ))
    print(
        string.format('First rougue = %.6f; Last rougue = %.6f' , 
            rouguef1[1], rouguef1[nepochs]) 
        )
end

cmd = torch.CmdLine()
cmd:option('--fast', false, 'parameter to evaluate speed')
cmd:option('--n_samples', 100, 'Number of queries')
-- n_samples = 100000 will reproduce the speed numbers
cmd:option('--n_s', 5, 'Number of sentences')
cmd:option('--q_l', 5, 'Query length')
cmd:option('--k', 7, 'Number of samples to iterate over')
cmd:option('--a', 1, 'Number of samples to iterate over')
cmd:option('--b', 100, 'Number of samples to iterate over')
cmd:option('--embDim', 100, 'Number of samples to iterate over')
cmd:option('--nepochs', 100, 'Number of epochs')
cmd:option('--epsilon', 1, 'Random sampling rate')
cmd:option('--print', false, 'print performance')
cmd:text()
local opt = cmd:parse(arg or {})       --- stores the commands in opt.variable (e.g., opt.model)

-- Running the script
runSimulation(opt.n_samples, opt.n_s, opt.q_l, opt.k, opt.a, opt.b,
              opt.embDim, opt.fast, opt.nepochs, opt.epsilon, opt.print)

-- Notes
-- 2. Optimize using masklayer
-- 5. Test on the GPU
-- 7. Deploy the model in chunks?