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
cmd:option('--batch_size', 500, 'batch size of 500')
cmd:text()

--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext2.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'
query_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/queries_numtext.csv'
sent_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_sentence_numtext2.csv'

data_file = csvigo.load({path = aurora_fn, mode = "large"})
nugget_file = csvigo.load({path = nugget_fn, mode = "large"})
query_file =  csvigo.load({path = query_fn, mode = "large"})
sent_file =  csvigo.load({path = sent_fn, mode = "large"})

delta = 1./(opt.nepochs/opt.cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs

torch.manualSeed(420)

vocab_sized = getVocabSize(data_file)                       --- getting length of dictionary
vocab_sizeq = getVocabSize(query_file)                      --- getting length of dictionary
vocab_sizes = getVocabSize(sent_file)                      --- getting length of dictionary
vocab_size = math.max(vocab_sized, vocab_sizeq, vocab_sizes)

queries = grabNsamples(query_file, #query_file-1, nil)      --- Extracting all queries
nuggets = grabNsamples(nugget_file, #nugget_file-1, nil)    --- Extracting all samples
maxseqlend = getMaxseq(data_file)                             --- Extracting maximum sequence length
maxseqlenq = getMaxseq(query_file)                            --- Extracting maximum sequence length
maxseqlen = math.max(maxseqlenq, maxseqlend)

batch_model  = build_model(opt.model, vocab_size, opt.embed_dim, 1, opt.usecuda)
crit = nn.MSECriterion()

out = iterateModel( opt.batch_size, opt.nepochs, queries[3], 
                    data_file, sent_file, 
                    batch_model, crit, opt.epsilon, delta, 
                    maxseqlen, opt.base_explore_rate, opt.print_every, 
                    nuggets, opt.learning_rate, opt.rK, opt.usecuda)