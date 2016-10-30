require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'cunnx'

dofile("utils.lua")
dofile("model_utils.lua")

usecuda = true
model = 'lstm'
metric = 'recall'
batch_size = 2
embed_dim = 10
outputSize = 1
learning_rate = 0.00001
epsilon = 1
start_epsilon = epsilon
nepochs = 100
cuts = 50
gamma = 0.3
delta = 1./(nepochs/cuts) 
base_explore_rate = 0.1
nsims = 10

queries   = {0, 1, 4, 3}
sentences = {
            {0, 1, 3, 4}, 
            {7, 6, 5 ,8}, 
            {0, 2, 4, 3}, 
            {0, 0, 0, 0}, 
            {1, 4, 3, 2}, 
            {13, 14, 15, 16} 
        }
true_action_list = {
                    1, 
                    0, 
                    1, 
                    0, 
                    1, 
                    0
                }
--- Let's say these are the true ones we need to pick
nuggets = {sentences[1], sentences[3], sentences[5], {9, 10, 12, 11}} 

vocab_size = 16
-- vocab_size = 12
criterion = nn.MSECriterion()

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

fullpreds = {0, 0, 0, 0, 0, 0}
action_list = torch.totable(torch.zeros(#sentences))
--- Theoretical limit
predsummary = buildPredSummary(true_action_list, sentences, nil)
predsummary = predsummary[#predsummary]
-- score = rougeF1({predsummary}, nuggets)
best_score = eval_func({predsummary}, nuggets)
print(string.format('Best possible Score = %.6f', best_score))
print(true_action_list)

modzero = 0.
modbest = 0.
mod = 0.
for sims = 1, nsims do
-- while sims < nsims and epsilon=0
    torch.manualSeed(sims)
    FinalMLP  = build_model(model, vocab_size, embed_dim, outputSize, usecuda)
    for epoch = 1, nepochs do
        --- Score the model with random actions first
        yrouge = score_model(action_list, 
                    sentences,
                    nuggets,
                    0, 
                    0, 
                    metric)
        predsummary = buildPredSummary(action_list, sentences, nil)
        predsummary = predsummary[#predsummary]
        score = eval_func({predsummary}, nuggets)

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
            fullpreds[minibatch] = torch.totable(preds)[1]
        end

        summaries = padZeros(buildCurrentSummary(action_list, sentences, 4 * 4), 4 * 4)
        --- Backward step
        for minibatch = 1, #sentences do
            summary  = LongTensor({summaries[minibatch]}):t()
            sentence = LongTensor({sentences[minibatch]}):t()
            query = LongTensor({queries}):t()        
            if (minibatch) < #sentences then
                labels = Tensor({yrouge[minibatch] + gamma * yrouge[minibatch + 1] })
            else 
                labels = Tensor({yrouge[minibatch]})
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

        pmin = math.min(table.unpack(fullpreds))
        pmax = math.max(table.unpack(fullpreds))
        pmean = sumTable(fullpreds) / #yrouge

        -- print(string.format("Predicted {min = %.6f, mean = %.6f, max = %.6f}", pmin, pmean, pmax))    
        print(string.format("epoch = %i, epsilon = %.3f, rouge-%s = %.6f, loss = %.6f, action = {%i, %i, %i ,%i, %i, %i}, Predicted: {min=%.6f, mean=%.6f, max=%.6f}", 
                epoch, epsilon, metric, score, loss, action_list[1], action_list[2], action_list[3], action_list[4], action_list[5], action_list[6], pmin, pmean, pmax )
        )

        if (epsilon - delta) <= base_explore_rate then
            epsilon = base_explore_rate
        else 
            epsilon = epsilon - delta
        end
        if score == 0. and epoch > torch.round(nepochs/cuts) then 
            epsilon = start_epsilon/2.
        end
    end
    if score == 0 then
        modzero = modzero + 1
    elseif score == best_score then
        modbest  = modbest + 1
    else 
        mod = mod + 1
    end
end
print(mod/nsims, modzero/nsims, modbest/nsims)