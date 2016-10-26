require 'nn'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'cunnx'

function build_bowmlp(nn_vocab_module, embed_dim)
    local model = nn.Sequential()
    :add(nn_vocab_module)            -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, embed_dim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_lstm(nn_vocab_module, embed_dim)
    local model = nn.Sequential()
    :add(nn_vocab_module)            -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_model(model, vocab_size, embed_dim, outputSize, use_cuda)
    local nn_vocab = nn.LookupTableMaskZero(vocab_size, embed_dim)
    if model == 'bow' then
        print("Running BOW model")
        mod1 = build_bowmlp(nn_vocab, embed_dim)
        mod2 = build_bowmlp(nn_vocab, embed_dim)
        mod3 = build_bowmlp(nn_vocab, embed_dim)
    end
    if model == 'lstm' then         
        print("Running LSTM model")
        mod1 = build_lstm(nn_vocab, embed_dim)
        mod2 = build_lstm(nn_vocab, embed_dim)
        mod3 = build_lstm(nn_vocab, embed_dim)
    end

    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)

    local FinalMLP = nn.Sequential()
    :add(ParallelModel)
    :add(nn.JoinTable(2))
    :add(nn.Linear(embed_dim * 3, 2) )
    FinalMLP:add(nn.Max(1) )
    FinalMLP:add(nn.Tanh())

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end

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
    yrouge = torch.rand(2, 1)
    if use_cuda then
        return sentences, summary, query, yrouge:cuda()
    else
        return sentences, summary, query,  yrouge
    end 
end    

usecuda = true
model = 'lstm'

batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.2

sentences, summary, query, yrouge = build_data(usecuda)
FinalMLP  = build_model(model, vocab_size, embed_dim, outputSize, usecuda)

criterion = nn.MSECriterion():cuda()

print(sentences)
print(summary)
print(query)
print(yrouge)
print('sumval =', sentences[1]:sum())

for epoch=1, 100 do
    preds = FinalMLP:forward({sentences, summary, query})
    loss = criterion:forward(preds, yrouge)
    grads = criterion:backward(preds, yrouge)
    FinalMLP:backward({sentences, summary, query}, grads)
    FinalMLP:updateParameters(learning_rate)
    FinalMLP:zeroGradParameters()
    if (epoch % 10)==0 then 
        print(string.format("Epoch %i, loss =%6f", epoch, loss))
    end
end