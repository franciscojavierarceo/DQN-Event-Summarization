require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'
m = csvigo.load({path = aurora_fn, mode = "large"})
q = csvigo.load({path = nugget_fn, mode = "large"})

N = 1000   --- #m-1
K = 100
torch.manualSeed(69)

out  = grabNsamples(m, N, K)            --- Extracting N samples
nggs = grabNsamples(q, #q-1, nil)       --- Extracting all samples
mxl  = getMaxseq(m)                     --- Extracting maximum sequence length
vocab_size = getVocabSize(out, N)       --- getting the length of the dictionary

xs = padZeros(out, mxl)             --- Padding the data by the maximum length
input = torch.LongTensor(xs)        --- This is the correct format to input it
labels = torch.round(torch.rand(#out))

preds = {}
for i=1,labels:size()[1] do
    preds[i] = labels[i]
end

predsummary = buildPredSummary(preds, xs)
rscore = rougeRecall(predsummary, nggs)
pscore = rougePrecision(predsummary, nggs)
fscore = rougeF1(predsummary, nggs)

--- Outputting the last rouge
perf_string = string.format("{Recall = %.6f, Precision = %.6f, F1 = %.6f}", rscore, pscore, fscore)
print(perf_string)


print("------------------")
print("  Model complete  ")
print("------------------")