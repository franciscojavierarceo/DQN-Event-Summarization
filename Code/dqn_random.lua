require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

--- Loading utility script
dofile("utils.lua")
dofile("model_utils.lua")

input_path = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/'

rK = 200
n = 10
aurora = {
        ['inputs'] = '2012_aurora_shooting_first_sentence_numtext2.csv', 
        ['nuggets'] = 'aurora_nuggets_numtext.csv',
        ['sentences'] = '2012_aurora_sentence_numtext2.csv'
}
pakistan = {
        ['inputs'] = '2012_pakistan_garment_factory_fires_first_sentence_numtext2.csv',
        ['nuggets'] ='pakistan_nuggets_numtext.csv',
        ['sentences'] = '2012_pakistan_sentence_numtext2.csv'
}

inputs = {
        aurora, 
        pakistan
    }

for query_id = 1, #inputs do
    input_file = csvigo.load({path = input_path .. inputs[query_id]['inputs'], mode = "large", verbose = false})
    nugget_file = csvigo.load({path = input_path .. inputs[query_id]['nuggets'], mode = "large", verbose = false})

    nuggets = grabNsamples(nugget_file, #nugget_file, nil)    --- Extracting all samples
    xs  = grabNsamples(input_file, 2, #input_file)

    rs, ps, fs = 0., 0., 0.
    for i = 1, n do
        torch.manualSeed(690 + i)
        action_list = torch.totable(torch.round(torch.rand(#input_file)))
        action_list[1] = 0

        predsummary = buildPredSummary(action_list, xs, rK)

        rscore = rougeRecall(predsummary, nuggets, rK)
        pscore = rougePrecision(predsummary, nuggets, rK)
        fscore = rougeF1(predsummary, nuggets, rK)

        rs, ps, fs = rs + rscore, ps + pscore, fs + fscore
        --- Outputting the last rougue
        perf_string = string.format(
            "sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                        sumTable(action_list), #action_list, rscore, pscore, fscore
        )
        print(perf_string)
    end

end
den = n * #inputs

perf_string = string.format(
    "Average {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                rs/den, ps/den, fs/den
)
print(perf_string)

print("------------------")
print("  Model complete  ")
print("------------------")