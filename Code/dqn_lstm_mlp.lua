require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'
m = csvigo.load({path = aurora_fn, mode = "large"})
q = csvigo.load({path = nugget_fn, mode = "large"})

N = 1000
K = 100
print_every = 10
nepochs = 100
embed_dim = 6
cuda = true
torch.manualSeed(69)
epsilon = 1.0
delta = 1./(nepochs/4.)

if N== nil then
    N = #m-1
end

function build_network(vocab_size, embed_dim, outputSize, cuda)
    batchLSTM = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(embed_dim, outputSize)) -- map last state to a score for classification
    :add(nn.ReLU())
   return batchLSTM
end

out = grabNsamples(m, N, K)             --- Extracting N samples
nggs = grabNsamples(q, #q-1, nil)       --- Extracting all samples

vocab_size = 0                          --- getting max length of vocab
for k,v in pairs(out) do
    vocab_size = math.max(vocab_size, math.max(table.unpack(v)))
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
end

mxl = 0
for k,v in pairs(out) do
    mxl = math.max(mxl, #v)
end

batchLSTM = build_network(vocab_size, embed_dim, 1, true)
crit = nn.MSECriterion()

xs = padZeros(out, mxl)             --- Padding the data by the maximum length
input = torch.LongTensor(xs)        --- This is the correct format to input it
labels = torch.randn(#out)          --- randn is from a normal whie rand() is uniform

-- For batch inputs, it's a little easier to start with 
-- (sequence-length x batch-size) tensor so we transpose the data

--- TO DO:
--- Need to modify the rouge scoring so that we are calcuating everything cumulatively
--- Then we know for each sentence whether or not it improves rouge
--- then we can start to map that as our label

myDataT = input:t()
loss = 0
for epoch=1, nepochs, 1 do
    myPreds = batchLSTM:forward(myDataT)
    loss = loss + crit:forward(myPreds, labels)
    grads = crit:backward(myPreds, labels)
    batchLSTM:backward(myDataT, grads)
    
    --We update params at the end of each batch
    batchLSTM:updateParameters(0.1)
    batchLSTM:zeroGradParameters()
    
    --- TO DO add the epsilon greedy strategy
    ---  to randomly choose based some epsilon% of the time
    preds = {}
    -- if 1 == 0 then
    if torch.rand(1)[1] <= epsilon then
        for i=1, N do
            preds[i] = (torch.rand(1)[1] > 0.5 ) and 1 or 0
        end
    else 
        --- This is the action choice 1 select, 0 skip
        for i=1, N do
            preds[i] = (myPreds[i][1] > 0) and 1 or 0
        end
    end

    --- Concatenating predictions into a summary
    predsummary = buildPredSummary(preds, xs)
    
    --- Initializing rouge metrics at time {t-1} and save scores
    rscores, pscores, fscores = {}, {}, {}
    r_t1 , p_t1, f_t1 = 0., 0., 0.      
    for i=1, N do
        --- Calculating rouge scores; Call get_i_n() to cumulatively computing rouge
        rscores[i] = rougeRecall(geti_n(predsummary, i), nggs) - r_t1
        pscores[i] = rougePrecision(geti_n(predsummary, i), nggs) - p_t1
        fscores[i] = rougeF1(geti_n(predsummary, i), nggs) - f_t1
        r_t1 = rscores[i]
        p_t1 = pscores[i]
        f_t1 = fscores[i]
    end
    rscore = table.unpack(rscores) / #rscores
    pscore = table.unpack(pscores) / #pscores
    fscore = table.unpack(fscores) / #fscores

    if (epoch%print_every)==0 then
        print(string.format("Epoch %i, {Recall = %.6f, Precision = %6.f, F1 = %.6f}", epoch, rscore, pscore, fscore))
    end
    --- Updating the inputs
    labels = torch.Tensor(fscores)
    epsilon = epsilon - delta
end

print("------------------")
print("  Model complete  ")
print("------------------")