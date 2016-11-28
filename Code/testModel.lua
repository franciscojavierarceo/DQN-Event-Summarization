require 'optim'
require 'io'
require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'cunnx'

dl = require 'dataload'
cmd = torch.CmdLine()

cmd:option('--nepochs', 5, 'running for 50 epochs')
cmd:option('--learning_rate', 1e-5, 'using a learning rate of 1e-5')
cmd:option('--gamma', 0., 'Discount rate parameter in backprop step')
cmd:option('--cuts', 4, 'Discount rate parameter in backprop step')
cmd:option('--base_explore_rate', 0.0, 'Base rate')
cmd:option('--mem_size', 100, 'Memory size')
cmd:option('--batch_size', 200,'Batch Size')
cmd:option('--model','bow','BOW/LSTM option')
cmd:option('--edim', 64,'Embedding dimension')
cmd:option('--usecuda', false, 'running on cuda')
cmd:option('--metric', "f1", 'Metric to learn')
cmd:option('--n_samples', 500, 'Number of samples to use')
cmd:option('--max_summary', 300, 'Maximum summary size')
cmd:option('--end_baserate', 5, 'Epoch number at which the base_rate ends')
cmd:option('--K_tokens', 25, 'Maximum number of tokens for each sentence')
cmd:option('--thresh', 0, 'Threshold operator')
cmd:option('--n_backprops', 1, 'Number of times to backprop through the data')
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
nnmod = opt.model
embeddingSize = opt.edim
use_cuda = opt.usecuda
metric = opt.metric
maxSummarySize = opt.max_summary
end_baserate = opt.end_baserate
n = opt.n_samples
K_tokens = opt.K_tokens
thresh = opt.thresh
n_backprops = opt.n_backprops

SKIP = 1
SELECT = 2
export = true
local epsilon = 1.0

local optimParams = {
    learningRate = opt.learning_rate,
}

dofile("utils.lua")
dofile("model_utils.lua")
dofile("model_utils2.lua")

input_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'
query_fn = input_path .. 'queries_numtext.csv'
query_file =  csvigo.load({path = query_fn, mode = "large", verbose = false})
queries = padZeros(buildTermDocumentTable(query_file, nil), 5)

torch.manualSeed(420)
math.randomseed(420)

pakistan = {
        ['inputs'] = '2012_pakistan_garment_factory_fires_first_sentence_numtext2.csv',
        ['nuggets'] ='pakistan_nuggets_numtext.csv',
        ['query'] = queries[2],
        ['query_name'] = 'pakistan'
}
aurora = {
        ['inputs'] = '2012_aurora_shooting_first_sentence_numtext2.csv', 
        ['nuggets'] = 'aurora_nuggets_numtext.csv',
        ['query'] = queries[3],
        ['query_name'] = 'aurora'
}
sandy = {
        ['inputs'] = 'hurricane_sandy_first_sentence_numtext2.csv',
        ['nuggets'] ='sandy_nuggets_numtext.csv',
        ['query'] = queries[7],
        ['query_name'] = 'sandy'
}

inputs = {
        aurora, 
        pakistan,
        sandy
    }

if use_cuda then
    Tensor = torch.CudaTensor
    LongTensor = torch.CudaLongTensor
    ByteTensor = torch.CudaByteTensor
    print("...running on GPU")
else
    torch.setnumthreads(8)
    Tensor = torch.Tensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor
    print("...running on CPU")
end

-- Initializing the model variables
vocabSize, query_data = intialize_variables(query_file, inputs, 
                                            input_path, K_tokens, 
                                            maxSummarySize)

local model = buildModel(nnmod, vocabSize, embeddingSize, use_cuda)
params, gradParams = model:getParameters()
criterion = nn.MSECriterion()

if use_cuda then
    criterion = criterion:cuda()
    model = model:cuda()
end

local perf = io.open(string.format("%s_perf.txt", nnmod), 'w')
for epoch=0, nepochs do
    for query_id=1, #inputs do
        -- Initializing local variables
            -- in principal I could use the query_data[query_id][1] 
            -- as the argument but that's much less readable
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
        local oracleF1 = query_data[query_id][14]
        exploreDraws:uniform(0, 1)

        -- Clearing out the variables from the last time

        for i=1, streamSize do      -- Iterating through individual sentences
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
            
            summary = buildSummary(
                actions:narrow(1, 1, i), 
                sentenceStream:narrow(1, 1, i),
                summaryBuffer:narrow(1, i + 1, 1),
                use_cuda
            )

            summaryOpt = buildSummary(
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
        --- Storing the data
        local input = {sentenceStream, queryBatch, summaryBatch}
        local memory = {input, reward, actions}

        if epoch == 0 and query_id == 1 then
            fullmemory = memory 
            randomF1 = f1
        else
            fullmemory = buildMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
            -- fullmemory = buildMemoryOld(memory, fullmemory, mem_size, batch_size, use_cuda)
        end
        --- Running backprop
        -- loss = backPropOld(memory, params, model, criterion, batch_size, mem_size, use_cuda)
        loss = backProp(memory, params, gradParams, optimParams, model, criterion, batch_size, n_backprops, use_cuda)

        if epoch == 0 and query_id == 1 then
            out = string.format("epoch;epsilon;loss;randomF1;oracleF1;rougeF1;rougeRecall;rougePrecision;actual;pred;nselect;nskip;query\n")
            perf:write(out)
        end
        nactions = torch.totable(actions:sum(1))[1]
        out = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}; %i; %i; %i\n", 
            epoch, epsilon, loss, randomF1, oracleF1, rougeF1, rougeRecall, rougePrecision,
            reward:min(), reward:max(),
            qValues:min(), qValues:max(),
            nactions[SELECT], nactions[SKIP],
            query_id
        )
        perf:write(out)

        local avpfile = io.open(string.format("plotdata/%s/%i/%i_epoch.txt", nnmod, query_id, epoch), 'w')
        avpfile:write("predSkip;predSelect;actual;Skip;Select;query\n")
        for i=1, streamSize do
            avpfile:write(string.format("%.6f;%.6f;%6f;%i;%i;%i\n", 
                    qValues[i][SKIP], qValues[i][SELECT], rouge[i], 
                    actions[i][SKIP], actions[i][SELECT], query_id))
        end
        avpfile:close()

        query_data[query_id] = {
            sentenceStream,
            streamSize,
            query,
            actions:fill(0),
            exploreDraws,
            summaryBuffer:zero(),
            qValues:zero(),
            rouge:zero(),
            actionsOpt:zero(),
            rougeOpt:zero(),
            refSummary,
            refCounts,
            buffer:zero(),
            oracleF1
        }

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
print(string.format("Model complete {Selected = %i; Skipped  = %i}; Final Rouge Recall, Precision, F1 = {%.6f;%.6f;%.6f}", nactions[SELECT], nactions[SKIP], rougeRecall, rougePrecision, rougeF1))
-- os.execute(string.format("python make_density_gif.py %i %s %s", nepochs, nnmod, metric))