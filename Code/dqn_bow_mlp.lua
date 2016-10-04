require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'
query_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/queries_numtext.csv'

data_file = csvigo.load({path = aurora_fn, mode = "large"})
nugget_file = csvigo.load({path = nugget_fn, mode = "large"})
query_file =  csvigo.load({path = query_fn, mode = "large"})

rK = 100
batch_size = 1000
nepochs = 10
print_every = 1
embed_dim = 10
learning_rate = 0.1

cuts = 4.                  --- This is the number of cuts we want
epsilon = 1.
base_explore_rate = 0.1
delta = 1./(nepochs/cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs

cuda = true
torch.manualSeed(420)

function build_network(vocab_size, embed_dim)
    model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, embed_dim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    -- :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_model(vocab_size, embed_dim, outputSize)
    mod1 = build_network(vocab_size, embed_dim)
    mod2 = build_network(vocab_size, embed_dim)
    mod3 = build_network(vocab_size, embed_dim)

    mlp1 = nn.Sequential()
    mlp1:add(nn.Linear(1, embed_dim))
    mlp1:add(nn.ReLU())

    ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)
    ParallelModel:add(mlp1)

    FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add( nn.Linear(embed_dim * 4, 1) )
    return FinalMLP
end

vocab_sized = getVocabSize(data_file)                       --- getting length of dictionary
vocab_sizeq = getVocabSize(query_file)                      --- getting length of dictionary
vocab_size = math.max(vocab_sized, vocab_sizeq)

queries = grabNsamples(query_file, #query_file-1, nil)      --- Extracting all queries
nuggets = grabNsamples(nugget_file, #nugget_file-1, nil)    --- Extracting all samples
maxlend = getMaxseq(data_file)                             --- Extracting maximum sequence length
maxlenq = getMaxseq(query_file)                            --- Extracting maximum sequence length
maxlen = math.max(maxlenq, maxlend)

batchMLP = build_network(vocab_size, embed_dim, 1)
crit = nn.MSECriterion()

out = iterateModel( batch_size, nepochs, queries[3], data_file, batchMLP, crit, epsilon, delta, 
                    maxlen, base_explore_rate, print_every, nuggets, learning_rate, rK)

print("------------------")
print("  Model complete  ")
print("------------------")