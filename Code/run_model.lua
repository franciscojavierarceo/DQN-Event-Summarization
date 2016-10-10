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
cmd:option('--K_tokens', 10, 'using last 200 sentences to calculate rougue')
cmd:option('--K_sentences', 10, 'using last 10 sentences to calculate rougue')
cmd:option('--batch_size', 500, 'batch size of 500')
cmd:option('--thresh', 0.05, 'rougue improvement threshold')
cmd:option('--embed_dim', 10, 'using an embedding dimension of 10')
cmd:option('--learning_rate', 0.01, 'using a learning rate of 0.01')
cmd:option('--print_every', 1, 'printing every 1 epoch')
cmd:option('--usecuda', true, 'running on cuda')
cmd:option('--epsilon', 0, 'starting with epsilon = 1')
cmd:option('--cuts', 4, 'using epsilon-greedy strategy 1/4 of the time')
cmd:option('--base_explore_rate', 0.25, 'base exploration rate of 0.25')
cmd:text()

--- this retrieves the commands and stores them in opt.variable (e.g., opt.model)
local opt = cmd:parse(arg or {})

print(opt.usecuda)

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

main_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'
aurora_fn = main_path .. '2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = main_path ..'aurora_nuggets_numtext.csv'
query_fn = main_path .. 'queries_numtext.csv'
sent_fn = main_path .. '2012_aurora_sentence_numtext.csv'

input_file = csvigo.load({path = aurora_fn, mode = "large", verbose = false})
nugget_file = csvigo.load({path = nugget_fn, mode = "large", verbose = false})
query_file =  csvigo.load({path = query_fn, mode = "large", verbose = false})
-- sent_file =  csvigo.load({path = sent_fn, mode = "large", verbose = false})

delta = 1./(opt.nepochs/opt.cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs

torch.manualSeed(420)

vocab_sized = getVocabSize(input_file)                       --- getting length of dictionary
vocab_sizeq = getVocabSize(query_file)                      --- getting length of dictionary
-- vocab_sizes = getVocabSize(sent_file)                      --- getting length of dictionary
vocab_size = math.max(vocab_sized, vocab_sizeq)

data_file = grabNsamples(input_file, #input_file, nil) 
queries = grabNsamples(query_file, #query_file, nil)      --- Extracting all queries
nuggets = grabNsamples(nugget_file, #nugget_file, nil)    --- Extracting all samples

maxseqlend = getMaxseq(input_file)                             --- Extracting maximum sequence length
maxseqlenq = getMaxseq(query_file)                            --- Extracting maximum sequence length
maxseqlen = math.max(maxseqlenq, maxseqlend)

batch_model  = build_model(opt.model, vocab_size, opt.embed_dim, 1, opt.usecuda)
crit = nn.MSECriterion()

out = iterateModel( opt.batch_size, opt.nepochs, queries[3], input_file, 
                    maxseqlen, batch_model, crit, opt.thresh, opt.epsilon, 
                    delta, opt.base_explore_rate, opt.print_every,  nuggets, 
                    opt.learning_rate, opt.K_tokens, opt.K_sentences, opt.usecuda)