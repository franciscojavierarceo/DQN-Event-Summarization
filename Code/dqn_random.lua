require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'

x = csvigo.load({path = aurora_fn, mode = "large"})
nugget_file = csvigo.load({path = nugget_fn, mode = "large"})

rK = 200
K = 200

for i = 1, 10 do
    torch.manualSeed(690 + i)

    nuggets = grabNsamples(nugget_file, #nugget_file-1, nil)
    xs  = grabNsamples(x, 1, #x)

    preds = torch.round(torch.rand(#xs))
    predsummary = buildPredSummary(preds, xs)

    rscore = rougeRecall(predsummary, nuggets, K)
    pscore = rougePrecision(predsummary, nuggets, K)
    fscore = rougeF1(predsummary, nuggets, K)

    --- Outputting the last rougue
    perf_string = string.format(
        "{Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                    rscore, pscore, fscore
    )

    print(perf_string)
end

print("------------------")
print("  Model complete  ")
print("------------------")