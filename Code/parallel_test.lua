require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'cunnx'

function build_mlp(vocab_size, embed_dim)
    local model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.ReLU())
   return model
end

function buildLSTM(vocab_size, embed_dim)
    local lstm = nn.Sequential()
    lstm:add(nn.LookupTableMaskZero(vsize, edim))
    lstm:add(nn.SplitTable(1, edim))
    lstm:add(nn.Sequencer(nn.LSTM(edim, edim)))
    lstm:add(nn.SelectTable(-1))
    return lstm
end

function buildFullModel(vocab_size, embed_dim, model, use_cuda)
    if model == 'lstm' then
        mod1 = build_mlp(vocab_size, embed_dim, usecuda)
        mod2 = build_mlp(vocab_size, embed_dim, usecuda)
        mod3 = build_mlp(vocab_size, embed_dim, usecuda)
    else         
        mod1 = buildLSTM(vocab_size, embed_dim, usecuda)
        mod2 = buildLSTM(vocab_size, embed_dim, usecuda)
        mod3 = buildLSTM(vocab_size, embed_dim, usecuda)
    end

    mod4 = nn.Sequential()
    mod4:add(nn.Linear(1, embed_dim))

    ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)
    ParallelModel:add(mod4)

    FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add( nn.Linear(embed_dim * 4, 1) )
    FinalMLP = FinalMLP
    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end 

end

usecuda = true
model = 'lstm'

batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.1

function build_data(use_cuda)    
    if use_cuda then
      Tensor = torch.CudaTensor
      LongTensor = torch.CudaLongTensor
    else
      Tensor = torch.Tensor
      LongTensor = torch.LongTensor
    end
    sentences = LongTensor{{0, 1, 3, 4}, {0, 2, 4, 3}}:t()
    summary = LongTensor{{0, 0, 1, 4}, {0, 2, 3, 1}}:t()
    query = LongTensor{{0, 0, 4, 3}, {0, 0, 0, 0}}:t()
    actions = torch.round(torch.rand(2, 1))
    yrouge = torch.rand(2)
    if use_cuda then
        return sentences, summary, query, actions:cuda(), yrouge:cuda()
    else
        return sentences, summary, query, actions, yrouge
    end 
end    

sentences, summary, query, actions, yrouge = build_data(usecuda)
FinalMLP  = buildFullModel(vocab_size, embed_dim, model, usecuda)

criterion = nn.MSECriterion():cuda()

print(sentences)
print(summary)
print(query)
-- actions:resize(2,1)
print(actions)

print('sumval =', sentences[1]:sum())

for epoch=1, 100 do
    preds = FinalMLP:forward({sentences, summary, query, actions})
    loss = criterion:forward(preds, yrouge)
    -- This is where it fails
    grads = criterion:backward(preds, yrouge)
    FinalMLP:backward({sentences, summary, query, actions}, grads)
    FinalMLP:updateParameters(learning_rate)
    FinalMLP:zeroGradParameters()
    if (epoch % 10)==0 then 
        print(string.format("Epoch %i, loss =%6f", epoch, loss))
    end
end

print(sentences)