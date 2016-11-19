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
cmd:option('--gamma', 0.4, 'Discount rate parameter in backprop step')
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
        -- pakistan,
        -- sandy
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

vocabSize = 0
maxseqlen = 0
maxseqlenq = getMaxseq(query_file)
vocabSizeq = getVocabSize(query_file)
query_data = {}
for query_id = 1, #inputs do
    input_fn = inputs[query_id]['inputs']
    nugget_fn = inputs[query_id]['nuggets']

    input_file = csvigo.load({path = input_path .. input_fn, mode = "large", verbose = false})
    nugget_file = csvigo.load({path = input_path .. nugget_fn, mode = "large", verbose = false})
    -- input_file = geti_n(input_file, 2, #input_file) 
    input_file = geti_n(input_file, 2, n) 
    nugget_file = geti_n(nugget_file, 2, #nugget_file) 

    vocabSize = math.max(vocabSize, vocabSizeq, getVocabSize(input_file))
    maxseqlen = math.max(maxseqlen, maxseqlenq, getMaxseq(input_file))

    action_list = torch.totable(torch.round(torch.rand(#input_file)))

    xtdm  = buildTermDocumentTable(input_file, K_tokens)
    nuggets = buildTermDocumentTable(nugget_file, nil)
    ntdm = {}
    for i=1, #nuggets do
        ntdm = tableConcat(table.unpack(nuggets), ntdm)
    end
    -- Initializing the bookkeeping variables and storing them
    query = LongTensor{inputs[query_id]['query'] }
    sentenceStream = LongTensor(padZeros(xtdm, K_tokens))
    streamSize = sentenceStream:size(1)
    refSummary = Tensor{ntdm}
    refCounts = buildTokenCounts(refSummary)
    buffer = Tensor(1, maxSummarySize):zero()
    actions = ByteTensor(streamSize, 2):fill(0)
    exploreDraws = Tensor(streamSize)
    summaryBuffer = LongTensor(streamSize + 1, maxSummarySize):zero()
    qValues = Tensor(streamSize, 2):zero()
    rouge = Tensor(streamSize + 1):zero()
    rouge[1] = 0
    summary = summaryBuffer:zero():narrow(1,1,1)
    
    query_data[query_id] = {
        query,
        sentenceStream,
        streamSize,
        refSummary,
        refCounts,
        buffer,
        actions,
        exploreDraws,
        summaryBuffer,
        qValues,
        rouge,
        summary
    }
end

model = buildModel(nnmod, vocabSize, embeddingSize, use_cuda)
params, gradParams = model:getParameters()
criterion = nn.MSECriterion()

if use_cuda then
    criterion = criterion:cuda()
    model = model:cuda()
end

local perf = io.open(string.format("%s_perf.txt", nnmod), 'w')
for epoch=0, nepochs do
    for query_id=1, #inputs do
        query = query_data[query_id][1]
        sentenceStream = query_data[query_id][2]
        streamSize = query_data[query_id][3]
        refSummary = query_data[query_id][4]
        refCounts = query_data[query_id][5]
        buffer = query_data[query_id][6]
        actions = query_data[query_id][7]
        exploreDraws = query_data[query_id][8]
        summaryBuffer = query_data[query_id][9]
        qValues = query_data[query_id][10]
        rouge = query_data[query_id][11]
        summary = query_data[query_id][12]

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
                summaryBuffer:narrow(1, i + 1, 1),
                use_cuda
                )

            local generatedCounts = buildTokenCounts(summary) 
            local recall, prec, f1 = rougeScores(generatedCounts, refCounts)

            if metric == "f1" then
                rouge[i + 1]  = threshold(f1, thresh)
            elseif metric == "recall" then
                rouge[i + 1]  = threshold(recall, thresh)
            elseif metric == "precision" then
                rouge[i + 1] = threshold(prec, thresh)
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
        local reward = reward0 + reward_tp1
        
        local querySize = query:size(2)
        local summaryBatch = summaryBuffer:narrow(1, 1, streamSize)
        local queryBatch = query:view(1, querySize):expand(streamSize, querySize) 
        local input = {sentenceStream, queryBatch, summaryBatch}
        --- Storing the data
        local memory = {input, reward, actions}

        if epoch == 0 then
            fullmemory = memory 
        else
            fullmemory = buildMemory(memory, fullmemory, mem_size, batch_size, use_cuda)
        end
        --- Running backprop
        loss = backProp(memory, params, gradParams, optimParams, model, criterion, batch_size, n_backprops, use_cuda)

        if epoch==0 then
            out = string.format("epoch;epsilon;loss;rougeF1;rougeRecall;rougePrecision;actual;pred;nselect;nskip;query\n")
            perf:write(out)
        end
        nactions = torch.totable(actions:sum(1))[1]
        out = string.format("%i; %.3f; %.6f; %.6f; %.6f; %.6f; {min=%.3f, max=%.3f}; {min=%.3f, max=%.3f}; %i; %i; %s\n", 
            epoch, epsilon, loss, rougeF1, rougeRecall, rougePrecision,
            reward:min(), reward:max(),
            qValues:min(), qValues:max(),
            nactions[1], nactions[2],
            inputs[query_id]['query_name']
        )
        perf:write(out)

        if export then 
            local ofile = io.open(string.format("plotdata/%s/%i_epoch.txt", nnmod, epoch), 'w')
            ofile:write("predSkip;predSelect;actual;Skip;Select\n")
            for i=1, streamSize do
                ofile:write(string.format("%.6f;%.6f;%6f;%i;%i\n", 
                        qValues[i][SKIP], qValues[i][SELECT], rouge[i], 
                        actions[i][SKIP], actions[i][SELECT]))
            end
            ofile:close()
        end 

        query_data[query_id] = {
            query,
            sentenceStream,
            streamSize,
            refSummary,
            refCounts,
            buffer,
            actions:fill(0),
            exploreDraws,
            summaryBuffer,
            qValues,
            rouge,
            summary
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
print(string.format("Model complete {Selected = %i; Skipped  = %i}; Final Rouge-F1 = {%.6f}", nactions[1], nactions[2], rougeF1))

-- os.execute(string.format("python make_density_gif.py %i %s %s", nepochs, nnmod, metric))