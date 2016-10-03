require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'

m = csvigo.load({path = aurora_fn, mode = "large"})
q = csvigo.load({path = nugget_fn, mode = "large"})

K = 100
rK = 100

nbatches = 10
nepochs = 100
print_every = 10
embed_dim = 6
learning_rate = 0.1

cuts = 4.                  --- This is the number of cuts we want
epsilon = 1.
base_explore_rate = 0.1
delta = 1./(nepochs/cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs

cuda = true
torch.manualSeed(420)

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

vocab_size = getVocabSize(m)            --- getting the length of the dictionary
nggs = grabNsamples(q, #q-1, nil)       --- Extracting all samples
mxl  = getMaxseq(m)                     --- Extracting maximum sequence length

batchLSTM = build_network(vocab_size, embed_dim, 1, true)
crit = nn.MSECriterion()

out = iterateModel(nbatches, nepochs, m, batchLSTM, crit, epsilon, delta, mxl,
                    base_explore_rate, print_every, nggs, learning_rate, rK)


print("------------------")
print("  Model complete  ")
print("------------------")