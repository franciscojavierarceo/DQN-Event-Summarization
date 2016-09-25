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
labels = torch.rand(#out)

-- For batch inputs, it's a little easier to start with 
-- (sequence-length x batch-size) tensor so we transpose the data
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
    
    preds = {}
    for i=1, myPreds:size()[1] do
        preds[i] = (myPreds[i][1] > 0) and 1 or 0 --- lua is stupid
    end
    --- Concatenating predictions into a summary
    predsummary = buildPredSummary(preds, xs)
    --- Calculating rouge scores
    rscore = rougeRecall(predsummary, nggs)
    pscore = rougePrecision(predsummary, nggs)
    fscore = rougeF1(predsummary, nggs)

    if (epoch%print_every)==0 then
        print(string.format("Epoch %i, Rouge \t {Recall = %.6f, Precision = %6.f, F1 = %.6f}", epoch, rscore, pscore, fscore))
    end
end

print("------------------")
print("  Model complete  ")
print("------------------")