require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

aroraname = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nuggets = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'

m = csvigo.load({path = aroraname, mode = "large"})
q = csvigo.load({path = nuggets, mode = "large"})

N = 1000 --- Breaks at 35
K = 100
embed_dim = 6
cuda = true
torch.manualSeed(420)

dofile("utils.lua")

--- Extracting N samples
out = grabNsamples(m, N, K)
nuggs = grabNsamples(q, #q-1, nil)

mxl = 0
for k,v in pairs(out) do
    mxl = math.max(mxl, #v)
end

--- getting the length of the dictionary
vocab_size = 0
for k,v in pairs(out) do
    vocab_size = math.max(vocab_size, math.max(table.unpack(v)))
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
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

xs = padZeros(out, mxl)             --- Padding the data by the maximum length
input = torch.LongTensor(xs)        --- This is the correct format to input it
labels = torch.rand(#out)

-- For batch inputs, it's a little easier to start with sequence-length x batch-size tensor, so we transpose songData
myDataT = input:t()
batchLSTM = build_network(vocab_size, embed_dim, 1, true)
crit = nn.MSECriterion()

loss = 0 
for i=1, 100, 1 do
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

    ys = unpackZeros(preds)
    predsummary = buildPredSummary(ys, xs)

    rscore = rougeRecall(predsummary, nuggs)
    pscore = rougePrecision(predsummary, nuggs)
    fscore = rougeF1(predsummary, nuggs)
    print(string.format("Iteration %i, Rouge \t {Recall = %.6f, Precision = %6.f, F1 = %.6f}", i, rscore, pscore, fscore))
end

--- Unpacking predictions and concatenating predictions into a summary