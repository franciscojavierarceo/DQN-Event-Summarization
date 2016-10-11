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
cmd:option('--nepochs', 50, 'running for 50 epochs')
cmd:option('--K_tokens', 10, 'using the first 10 tokens to extract data')
cmd:option('--K_sentences', 20, 'using last 10 sentences to calculate rougue')
cmd:option('--batch_size', 500, 'batch size of 500')
cmd:option('--thresh', 0.01, 'rougue improvement threshold')
cmd:option('--embed_dim', 10, 'using an embedding dimension of 10')
cmd:option('--learning_rate', 0.01, 'using a learning rate of 0.01')
cmd:option('--print_every', 1, 'printing every 1 epoch')
cmd:option('--usecuda', true, 'running on cuda')
cmd:option('--epsilon', 1, 'starting with epsilon = 1')
cmd:option('--cuts', 4, 'using epsilon-greedy strategy 1/4 of the time')
cmd:option('--base_explore_rate', 0.25, 'base exploration rate of 0.25')
cmd:text()

--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

torch.manualSeed(420)

data_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'

query_fn = data_path .. 'queries_numtext.csv'
query_file =  csvigo.load({path = query_fn, mode = "large", verbose = false})
queries = buildTermDocumentTable(query_file, nil)

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
--- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs
delta = 1./(opt.nepochs/opt.cuts) 
crit = nn.MSECriterion()


out = iterateModelQueries(data_path, query_file, opt.batch_size, opt.nepochs, inputs, 
                            opt.model, crit, opt.thresh, opt.embed_dim, opt.epsilon, delta, 
                            opt.base_explore_rate, opt.print_every,
                            opt.learning_rate, opt.K_tokens, opt.K_sentences, opt.usecuda)