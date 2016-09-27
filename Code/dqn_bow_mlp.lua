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

N = 1000
K = 100
print_every = 10
nepochs = 100
embed_dim = 6
cuda = true
torch.manualSeed(420)
epsilon = 1.
cuts = 4.                  --- This is the number of cuts we want
delta = 1./(nepochs/cuts) --- Only using epsilon greedy strategy for (nepochs/cuts)% of the epochs
base_explore_rate = 0.1
learning_rate = 0.1

if N == nil then
    N = #m-1
end

function build_network(vocab_size, embed_dim, outputSize, cuda)
    bowMLP = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, embed_dim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Linear(embed_dim, outputSize)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return bowMLP
end

out  = grabNsamples(m, N, K)            --- Extracting N samples
nggs = grabNsamples(q, #q-1, nil)       --- Extracting all samples
mxl  = getMaxseq(m)                     --- Extracting maximum sequence length
vocab_size = getVocabSize(out, N)       --- getting the length of the dictionary

batchMLP = build_network(vocab_size, embed_dim, 1, true)
crit = nn.MSECriterion()

xs = padZeros(out, mxl)             --- Padding the data by the maximum length
input = torch.LongTensor(xs)        --- This is the correct format to input it
labels = torch.randn(#out, 0, 2)          --- randn is from a normal whie rand() is uniform
print(labels:min(), labels:max(), labels:median(), labels:mean())

-- For batch inputs, it's a little easier to start with 
-- (sequence-length x batch-size) tensor so we transpose the data
iterateModel(nepochs, batchMLP, crit, input, 
                    xs, labels, epsilon, delta, 
                    base_explore_rate, print_every,
                    nggs, learning_rate)

print("------------------")
print("  Model complete  ")
print("------------------")