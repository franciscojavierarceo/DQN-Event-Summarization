require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'cunnx'

dofile("utils.lua")
dofile("model_utils.lua")

usecuda = true
model = 'lstm'
batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.2
epsilon = 1
nepochs = 30
cuts = 2.
delta = 1./(nepochs/cuts) 
base_explore_rate = 0.1

FinalMLP  = build_model(model, vocab_size, embed_dim, outputSize, usecuda)
criterion = nn.MSECriterion()

sentences = {{0, 1, 3, 4}, {0, 0, 0 ,0}, {0, 2, 4, 3}, {1, 4, 3, 2} }
summaries = {{0, 0, 0, 0}, {0, 1, 3, 4}, {}, {} }
queries = {{0, 1, 4, 3}, {0, 1, 4, 3}, {0, 1, 4, 3}, {0, 1, 4, 3}}
--- Let's say these are the true ones we need to pick
nuggets = {sentences[1], sentences[3], sentences[4], {5, 7, 3, 1}} 

if usecuda then
  Tensor = torch.CudaTensor
  LongTensor = torch.CudaLongTensor
  criterion = criterion:cuda()
else
  Tensor = torch.Tensor
  LongTensor = torch.LongTensor
end


fullpreds = {0, 0, 0, 0}
action_list = {0, 0, 0, 0}
--- Theoretical limit
predsummary = buildPredSummary({1, 0, 1, 1}, sentences, nil)
predsummary = predsummary[#predsummary]
fscore = rougeF1({predsummary}, nuggets)
print(string.format('Best possible F1 = %.6f', fscore))

for epoch = 1, nepochs do
    --- Forward step
    for minibatch = 1, #sentences do
        summaries = padZeros(buildCurrentSummary(geti_n(action_list, 1, minibatch), 
                                               geti_n(sentences, 1, minibatch), nil), 9)

        summary  = LongTensor({summaries[minibatch]}):t()
        sentence = LongTensor({sentences[minibatch]}):t()
        query = LongTensor({queries[minibatch]}):t()
        -- forward pass
        preds = FinalMLP:forward({sentence, summary, query})
        pred_actions = torch.totable(FinalMLP:get(3).output)
        if torch.rand(1)[1] < epsilon then 
            opt_action = torch.round(torch.rand(1))[1]
        else 
            opt_action = (pred_actions[1][1] > pred_actions[1][2]) and 0 or 1
        end
        --- Updating bookeeping variables
        action_list[minibatch] = opt_action
        fullpreds[minibatch] = torch.totable(preds)
    end
    --- Score after forward pass
    scores = score_model(action_list, 
                sentences,
                nuggets,
                0, 
                0, 
                'f1')
    predsummary = buildPredSummary(action_list, sentences, nil)
    predsummary = predsummary[#predsummary]
    fscore = rougeF1({predsummary}, nuggets)

    --- Backward step
    for minibatch = 1, #sentences do
        summary  = LongTensor({summaries[minibatch]}):t()
        sentence = LongTensor({sentences[minibatch]}):t()
        query = LongTensor({queries[minibatch]}):t()        
        yrouge = Tensor({scores[minibatch]}):cuda()
        preds = Tensor(fullpreds[minibatch])
        --- Backward pass        
        preds = FinalMLP:forward({sentence, summary, query})
        loss = criterion:forward(preds, yrouge)
        FinalMLP:zeroGradParameters()
        grads = criterion:backward(preds, yrouge)
        FinalMLP:backward({sentence, summary, query}, grads)
        FinalMLP:updateParameters(learning_rate)
        -- if minibatch > 1 then
        --     print(epsilon, minibatch, fscore, table.unpack(summaries[minibatch]))
        -- end
    end
    print(epoch, fscore, table.unpack(unpackZeros(predsummary)))
    --- Random part
    if (epsilon - delta) <= base_explore_rate then
        epsilon = base_explore_rate
    else 
        epsilon = epsilon - delta
    end
end