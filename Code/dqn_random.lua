require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
nugget_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'

rK = 500
x = csvigo.load({path = aurora_fn, mode = "large"})
nugget_file = csvigo.load({path = nugget_fn, mode = "large"})

n = 10

nuggets = grabNsamples(nugget_file, #nugget_file-1, nil)    --- Extracting all samples
xs  = grabNsamples(x, 1, #x)

rs, ps, fs = 0., 0., 0.
for i = 1, n do
    torch.manualSeed(690 + i)

    preds = torch.round(torch.rand(#xs))
    predsummary = buildPredSummary(preds, xs)

    rscore = rougeRecall(predsummary, nuggets, rK)
    pscore = rougePrecision(predsummary, nuggets, rK)
    fscore = rougeF1(predsummary, nuggets, rK)
    rs, ps, fs = rs + rscore, ps + pscore, fs + fscore
    --- Outputting the last rougue
    perf_string = string.format(
        "{Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                    rscore, pscore, fscore
    )

    print(perf_string)
end
--- Outputting the last rougue
perf_string = string.format(
    "Average {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                rs/n, ps/n, fs/n
)

    print(perf_string)

print("------------------")
print("  Model complete  ")
print("------------------")