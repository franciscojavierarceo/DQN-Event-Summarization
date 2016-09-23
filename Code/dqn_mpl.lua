require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

local dl = require 'dataload'
datapath = '/Users/franciscojavierarceo/data/Twitter/'
trainSet, validSet, testSet = dl.loadSentiment140(datapath, minFreq,
                                                  seqLen, validRatio)

aroraname = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_numtext.csv'
wordfile = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/total_corpus_smry.csv'
queryfile = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/queries_numtext.csv'

m = csvigo.load({path = aroraname, mode = "large"})
w = csvigo.load({path = wordfile, mode = "large"})
q = csvigo.load({path = queryfile, mode = "large"})

N = 1000
K = 100
embed_dim = 6
 
dofile("utils.lua")

out = {}
for k,v in pairs(m) do
    if k > 1 then
        out[k-1] = grabKtokens(split(m[k][1]), K)
    end
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
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

local opt = {}

opt.hiddenSize = 10
opt.vocabSize = 8

x = torch.Tensor{{1, 2, 3, 4},
                {0, 5, 8, 3}}

local enc = nn.Sequential()
enc:add(nn.LookupTableMaskZero(opt.vocabSize, opt.hiddenSize))

layer1 = nn.SeqLSTM(opt.hiddenSize, opt.hiddenSize)
layer1:maskZero()
enc:add(layer1)


criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
--- Embedding layer

enc:add(nn.Select(1, -1))
--- Add mlp

print(enc:forward(x:t()))

---batch by embedding size is output
opt.learningRate = 0.01
opt.niter = 4 


for i=1,opt.niter do
   enc:zeroGradParameters()
   -- Forward pass
    encOut = enc:forward(x)
   --print(decOut)
    err = criterion:forward(encOut)
   
   print(string.format("Iteration %d ; NLL err = %f ", i, err))
   -- Backward pass
   gradOutput = criterion:backward(encOut)
   zeroTensor = torch.Tensor(encOut):zero()
   enc:backward(x, zeroTensor)

   dec:updateParameters(opt.learningRate)
   enc:updateParameters(opt.learningRate)
end

