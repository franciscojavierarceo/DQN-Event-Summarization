require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'
require 'cutorch'
require 'cunn'
require 'cunnx'

cmd = torch.CmdLine()
--- setting the parameter defaults
cmd:option('--model', 'lstm', 'using LSTM instead of BOW')                       
cmd:option('--rK', 200, 'using last 200 sentences to calculate rougue')
cmd:option('--nepochs', 50, 'running for 50 epochs')
cmd:option('--print_every', 1, 'printing every 1 epoch')
cmd:option('--embed_dim', 10, 'using an embedding dimension of 10')
cmd:option('--learning_rate', 0.01, 'using a learning rate of 0.01')
cmd:option('--usecuda', true, 'running on cuda')
cmd:option('--epsilon', 1, 'starting with epsilon = 1')
cmd:option('--cuts', 4, 'using epsilon-greedy strategy 1/4 of the time')
cmd:option('--base_explore_rate', 0.25, 'base exploration rate of 0.25')
cmd:option('--batch_size', 200, 'batch size of 500')
cmd:text()

--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

data_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'
query_fn = data_path .. 'queries_numtext.csv'
query_file =  csvigo.load({path = query_fn, mode = "large"})
queries = grabNsamples(query_file, #query_file, nil)

aurora = {
        ['inputs'] = '2012_aurora_shooting_first_sentence_numtext.csv', 
        ['nuggets'] = 'aurora_nuggets_numtext.csv',
        ['sentences'] = '2012_aurora_sentence_numtext.csv',
        ['query'] = queries[3]
}
pakistan = {
        ['inputs'] = '2012_pakistan_garment_factory_fires_numtext.csv',
        ['nuggets'] ='pakistan_nuggets_numtext.csv',
        ['sentences'] = '2012_aurora_sentence_numtext.csv',
        ['query'] = queries[2]
}

inputs = {
        aurora, 
        pakistan
    }
--- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs
torch.manualSeed(420)
delta = 1./(opt.nepochs/opt.cuts) 
crit = nn.MSECriterion()

out = iterateModelQueries(data_path, query_file, opt.batch_size, opt.nepochs, 
                            inputs, 
                            opt.model, crit, opt.embed_dim, opt.epsilon, delta, 
                            opt.base_explore_rate, opt.print_every,
                            opt.learning_rate, opt.rK, opt.usecuda)