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


function runSimulation(n, n_s, q, k, a, b, embDim)
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
    print("predictions = ")
    print(preds)
    -- Pulling the best actions
    qMax, qindx = torch.max(preds, 2)
    -- Here's the fast way to select the optimal action for each query
    actions = torch.zeros(n, 2):scatter(2, qindx, torch.ones(preds:size()))
    
    -- Here's the slow way to build the predicted summary...
        -- This is what I'd like to optimize further, right now it's a for loop
        -- it'd be great if we could skip the loop...maybe with a matrix multiplication to 
        -- wipe out all of the non-zero elements...
    predsummary = buildPredsummary(predsummary, actions, sentences[1], SELECT)
    print("predicted summary = ")
    print(predsummary)
end

-- Setting parameters
local n = 10
local n_s = 5
local k = 7
local q = 5
local a = 1
local b = 100
local embDim = 50

runSimulation(n, n_s, q, k, a, b, embDim)


