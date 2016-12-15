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

cmd:option('--nepochs', 10000, 'running for 50 epochs')
cmd:option('--learning_rate', 1e-4, 'using a learning rate of 1e-4')
cmd:option('--embeddingSize', 64,'Embedding dimension')
cmd:option('--gamma', 0.3, 'Discount rate parameter in backprop step')
cmd:option('--model','bow','BOW/LSTM option')
cmd:option('--cuts', 10, 'Discount rate parameter in backprop step')
cmd:option('--epsilon', 1., 'Random search rate')
cmd:option('--base_explore_rate', 0.1, 'Base rate')
cmd:option('--end_baserate', 100, 'Epoch number at which the base_rate ends')
cmd:option('--batch_size', 200,'Batch Size')
cmd:option('--mem_size', 3000, 'Memory size')
cmd:option('--metric', "f1", 'Metric to learn')
cmd:option('--n_samples', nil, 'Number of samples to use')
cmd:option('--maxSummarySize', 300, 'Maximum summary size')
cmd:option('--K_tokens', 25, 'Maximum number of tokens for each sentence')
cmd:option('--thresh', 0, 'Threshold operator')
cmd:option('--adapt', false, 'Domain Adaptation Regularization')
cmd:option('--datapath', 'data/0-output/', 'Path to input data')
cmd:option('--usecuda', false, 'running on cuda')
cmd:option('--n_backprops', 3, 'Number of times to backprop through the data')
cmd:text()
--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

dofile("Code/utils.lua")
dofile("Code/utilsNN.lua")

inputs = loadMetadata(opt.datapath .. "dqn_metadata.csv")
stopfile = csvigo.load({path = opt.datapath .. 'stopwordids.csv', mode = "large", verbose = false})
stoplist = buildTermDocumentTable(stopfile, nil)

stopwords = {}
for k,v in pairs(stoplist) do 
    stopwords[k]  = v[1]
end

if opt.usecuda then
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

local delta = opt.cuts/opt.nepochs
local optimParams = { learningRate = opt.learning_rate }

-- Initializing the model variables
local vocabSize, query_data = intialize_variables(inputs, 
                                            opt.n_samples, opt.datapath, opt.K_tokens, 
                                            opt.maxSummarySize, stopwords, opt.thresh, opt.usecuda)
local model = buildModel(opt.model, vocabSize, opt.embeddingSize, opt.metric, opt.adapt, opt.usecuda)

-- Running the model
trainCV(inputs, query_data, model, opt.nepochs, opt.model, opt.metric, opt.thresh, 
      opt.gamma, opt.epsilon, delta, opt.base_explore_rate, opt.end_baserate, 
      opt.mem_size, opt.batch_size, optimParams, opt.n_backprops, opt.adapt, 
      stopwords, opt.usecuda)

-- os.execute(string.format("python make_density_gif.py %i %s %s", nepochs, nnmod, metric))