require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'cunnx'

dofile("utils.lua")
dofile("model_utils.lua")

usecuda = true
model = 'lstm'
metric = 'f1'
batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.1
epsilon = 1
nepochs = 100
cuts = 2.
gamma = 0.3
delta = 1./(nepochs/cuts) 
base_explore_rate = 0.

FinalMLP  = build_model(model, vocab_size, embed_dim, outputSize, usecuda)
criterion = nn.MSECriterion()

sentences = {{0, 1, 3, 4}, {0, 0, 0 ,0}, {0, 2, 4, 3}, {1, 4, 3, 2} }
summaries = {{0, 0, 0, 0}, {0, 1, 3, 4}, {}, {} }
queries = {0, 1, 4, 3}
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

if metric=='f1' then
    eval_func = rougeF1
elseif metric=='recall' then
    eval_func = rougeRecall
elseif metric=='precision' then
    eval_func = rougePrecision
end

fullpreds = {0, 0, 0, 0}
action_list = {0, 0, 0, 0}
true_action_list = {1, 0, 1, 1}
--- Theoretical limit
predsummary = buildPredSummary({1, 0, 1, 1}, sentences, nil)
predsummary = predsummary[#predsummary]
-- score = rougeF1({predsummary}, nuggets)
score = eval_func({predsummary}, nuggets)
print(string.format('Best possible Score = %.6f', score))
print(true_action_list)

for epoch = 1, nepochs do
    --- Forward step
    for minibatch = 1, #sentences do
        summary, sentence, query = BuildTensors(action_list, sentences, queries, minibatch, 4, 4)
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
                metric)
    predsummary = buildPredSummary(action_list, sentences, nil)
    predsummary = predsummary[#predsummary]
    score = eval_func({predsummary}, nuggets)

    summaries = padZeros(buildCurrentSummary(action_list, sentences, 4 * 4), 4 * 4)
    --- Backward step
    for minibatch = 1, #sentences do
        summary  = LongTensor({summaries[minibatch]}):t()
        sentence = LongTensor({sentences[minibatch]}):t()
        query = LongTensor({queries}):t()        
        if (minibatch) < #sentences then
            labels = Tensor({scores[minibatch] + gamma * scores[minibatch + 1] })
        else 
            labels = Tensor({scores[minibatch]})
        end
        preds = Tensor(fullpreds[minibatch])
        --- Backward pass        
        preds = FinalMLP:forward({sentence, summary, query})
        loss = criterion:forward(preds, labels)
        FinalMLP:zeroGradParameters()
        grads = criterion:backward(preds, labels)
        FinalMLP:backward({sentence, summary, query}, grads)
        FinalMLP:updateParameters(learning_rate)
    end

    print_str = string.format("epoch = %i, epsilon = %.3f, rouge-%s = %.6f, loss = %.6f, action = {%i, %i, %i ,%i} ", 
            epoch, epsilon, metric, score, loss, action_list[1], action_list[2], action_list[3], action_list[4])
    print(print_str)
    if (epsilon - delta) <= base_explore_rate then
        epsilon = base_explore_rate
    else 
        epsilon = epsilon - delta
    end
end