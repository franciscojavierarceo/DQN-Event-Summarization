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
cmd:option('--epsilon', 1., 'Random search rate')
cmd:option('--cuts', 4, 'Discount rate parameter in backprop step')
cmd:option('--base_explore_rate', 0.0, 'Base rate')
cmd:option('--mem_size', 100, 'Memory size')
cmd:option('--batch_size', 200,'Batch Size')
cmd:option('--model','bow','BOW/LSTM option')
cmd:option('--embeddingSize', 64,'Embedding dimension')
cmd:option('--usecuda', false, 'running on cuda')
cmd:option('--metric', "f1", 'Metric to learn')
cmd:option('--n_samples', 500, 'Number of samples to use')
cmd:option('--maxSummarySize', 300, 'Maximum summary size')
cmd:option('--end_baserate', 5, 'Epoch number at which the base_rate ends')
cmd:option('--K_tokens', 25, 'Maximum number of tokens for each sentence')
cmd:option('--thresh', 0, 'Threshold operator')
cmd:option('--n_backprops', 3, 'Number of times to backprop through the data')
cmd:text()
--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

dofile("utils.lua")
dofile("utilsNN.lua")

input_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'
query_fn = input_path .. 'queries_numtext.csv'
query_file =  csvigo.load({path = query_fn, mode = "large", verbose = false})
queries = padZeros(buildTermDocumentTable(query_file, nil), 5)

local pakistan = {
        ['inputs'] = '2012_pakistan_garment_factory_fires_first_sentence_numtext2.csv',
        ['nuggets'] ='pakistan_nuggets_numtext.csv',
        ['query'] = queries[2],
        ['query_name'] = 'pakistan'
}
local aurora = {
        ['inputs'] = '2012_aurora_shooting_first_sentence_numtext2.csv', 
        ['nuggets'] = 'aurora_nuggets_numtext.csv',
        ['query'] = queries[3],
        ['query_name'] = 'aurora'
}
local sandy = {
        ['inputs'] = 'hurricane_sandy_first_sentence_numtext2.csv',
        ['nuggets'] ='sandy_nuggets_numtext.csv',
        ['query'] = queries[7],
        ['query_name'] = 'sandy'
}

local inputs = {
        aurora, 
        -- pakistan,
        -- sandy
}

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

local delta = 1./(opt.nepochs/opt.cuts) 
local optimParams = { learningRate = opt.learning_rate }

-- Initializing the model variables
local vocabSize, query_data = intialize_variables(query_file, inputs, 
                                            opt.n_samples, input_path, opt.K_tokens, 
                                            opt.maxSummarySize)
local model = buildModel(opt.model, vocabSize, opt.embeddingSize, opt.metric, opt.usecuda)

-- Running the model
train(inputs, query_data, model, opt.nepochs, opt.model, opt.metric, opt.thresh, 
      opt.gamma, opt.epsilon, delta, opt.base_explore_rate, opt.end_baserate, 
      opt.mem_size, opt.batch_size, optimParams, opt.n_backprops, opt.usecuda)

-- os.execute(string.format("python make_density_gif.py %i %s %s", nepochs, nnmod, metric))