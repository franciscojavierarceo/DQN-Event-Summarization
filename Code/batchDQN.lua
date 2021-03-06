require 'os'
require 'nn'
--require 'cunn'
--require 'cunnx'
require 'optim'

--require 'cudnn'
require 'rnn'
require 'csvigo'

dl = require 'dataload'

dofile("Code/Utils/load_cnn.lua")
dofile("Code/utilsNNbatch.lua")

Tensor = torch.Tensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

torch.setnumthreads(torch.getnumthreads())

outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training_ss/'
datafile = "cnn_data_ss.dat"

data = torch.load(outputpath .. datafile)
print("...data loaded")

queries = data[1]
trueSummaries = data[2]
sentences= data[3]

n = 1000

queries = queries[{{1, n}}]
trueSummaries = trueSummaries[{{1, n}}]

tmp = {}
for j=1,#sentences do 
    tmp[j] = sentences[j][{{1, n}}]
end

sentences = tmp

print(string.format("# of sentences = %i", #sentences))

cmd = torch.CmdLine()
cmd:option('--vocab_size', 20001, 'Number of samples to iterate over')
cmd:option('--lr', 1e-5, 'Learning rate')
cmd:option('--embDim', 50, 'Number of samples to iterate over')
cmd:option('--gamma', 0., 'Weight of future prediction')
cmd:option('--batch_size', 25, 'Batch size')
cmd:option('--memory_multiplier', 0.05, 'Multiplier defining size of memory')
cmd:option('--cuts', 4, 'How long we want to decay search over')
cmd:option('--endexplorerate', 0.8, 'When to end the exploration as a percent of trainings epochs)')
cmd:option('--base_explore_rate', 0.1, 'Base exploration rate after 1/cuts until endexplorerate')
cmd:option('--nepochs', 100, 'Number of epochs')
cmd:option('--epsilon', 1, 'Random sampling rate')
cmd:option('--print', false, 'print performance')
cmd:option('--adapt', false, 'Use adaptive regularization')
cmd:option('--adapt_lambda', 0.25, 'Amount of adaptive regularization')
cmd:option('--usecuda', false, 'cuda option')
cmd:option('--seedval', 420, 'seedvalue')
cmd:text()
local opt = cmd:parse(arg or {})       --- stores the commands in opt.variable (e.g., opt.model)

-- Running the script
train(
    queries, sentences, trueSummaries, opt.lr, opt.vocab_size, 
    opt.embDim, opt.gamma, opt.batch_size, opt.nepochs, opt.epsilon, opt.print, 
    opt.memory_multiplier, opt.cuts, opt.base_explore_rate, opt.endexplorerate, 
    opt.adapt, opt.adapt_lambda, opt.usecuda, opt.seedval
)
