require 'os'
require 'nn'
require 'rnn'

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
--     actionmatrix = chosenactions:select(2, select_index):resize(1, n):view(n, 1):expand(n, k):clone()
    return actionmatrix:cmul(inputsentences:double())
end

function buildTotalSummary(predsummary1, totalPredsummary)
    nps = predsummary1:size(1)
    n_l = totalPredsummary:size(2)
    indices = torch.linspace(1, n_l, n_l):long() 
    for i=1, predsummary1:size(1) do
        if predsummary1[i]:sum() > 0 then 
            -- Finding the largest index with a zero
            maxindex = torch.max(indices[torch.eq(totalPredsummary[i], 0)])
            lenx = predsummary1[i]:size(1)
            totalPredsummary[i][{{maxindex - lenx + 1, maxindex}}]:copy(predsummary1[i])
        end
    end
end

function runSimulation(n, n_s, q, k, a, b, embDim, fast)
    local SKIP = 1
    local SELECT = 2
    -- Simulating streams and queries
    queries = genNbyK(n, q, a, b)
    -- Note that the sentences are batched by sentence index so sentences[1] is the first sentence of each article
    sentences = {}
    for i=1, n_s do
        sentences[i] = genNbyK(n, k, a, b)
    end
    -- Using this to generate the optimal actions
    true_actions = {}
    for i=1, n_s do 
        ---- Simulating the data
        trueqValues = torch.rand(n, 2)                   
         ---- Generating the max values and getting the indices
        qMaxtrue, qindxtrue = torch.max(trueqValues, 2) 
        --- I want to select the qindx elements for each row
        true_actions[i] = torch.zeros(n, 2):scatter(2, qindxtrue, torch.ones(trueqValues:size()))
    end
    model = buildModel('bow', b, embDim, 'f1', false, false)
    preds = model:forward({sentences[1], queries, torch.zeros(n, q)})
    -- print("predictions = ")
    -- print(preds)
    -- Pulling the best actions
    qMax, qindx = torch.max(preds, 2)
    -- Here's the fast way to select the optimal action for each query
    actions = torch.zeros(n, 2):scatter(2, qindx, torch.ones(preds:size()))
    
    nClock = os.clock()
    totalpredsummary = {}
    for i = 1, #sentences do
        -- This one saves quite a bit of time...from ~0.16 seconds vs 3.34 seconds...21x faster
        if fast then 
            predsummary = buildPredsummaryFast(predsummary, actions, sentences[i], SELECT)
        else 
            predsummary = buildPredsummary(predsummary, actions, sentences[i], SELECT)
        end
        totalpredsummary[i] = predsummary
    end
    print(string.format("Elapsed time: %.5f" % (os.clock()-nClock) ))

end


cmd = torch.CmdLine()
cmd:option('--fast', false, 'parameter to evaluate speed')
cmd:option('--n_samples', 100, 'Number of queries')
-- n_samples = 100000 will reproduce the speed numbers
cmd:option('--n_s', 5, 'Number of sentences')
cmd:option('--q_l', 5, 'Query length')
cmd:option('--k', 7, 'Number of sampels to iterate over')
cmd:option('--a', 1, 'Number of sampels to iterate over')
cmd:option('--b', 100, 'Number of sampels to iterate over')
cmd:option('--embDim', 100, 'Number of sampels to iterate over')
cmd:text()
local opt = cmd:parse(arg or {})       --- stores the commands in opt.variable (e.g., opt.model)

-- Running the script
runSimulation(opt.n_samples, opt.n_s, opt.q_l, opt.k, opt.a, opt.b, opt.embDim, opt.fast)

