require 'nn'
require 'rnn'
require 'cutorch'

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

function buildLSTM(vsize, edim)
    local lstm = nn.Sequential()
    lstm:add(nn.LookupTableMaskZero(vsize, edim))
    lstm:add(nn.SplitTable(1, edim))
    lstm:add(nn.Sequencer(nn.LSTM(edim, edim)))
    lstm:add(nn.SelectTable(-1))
    return lstm
end

sentences = torch.LongTensor{{0, 1, 3, 4}, {0, 2, 4, 3}}:t()
summary = torch.LongTensor{{0, 0, 1, 4}, {0, 2, 3, 1}}:t()
query = torch.LongTensor{{0, 0, 4, 3}, {0, 0, 0, 0}}:t()

-- sentences = torch.LongTensor{{0, 1, 3, 4}, {0, 2, 4, 3}, {0, 0, 4, 3}}:t()
-- summary = torch.LongTensor{{0, 0, 1, 4}, {0, 2, 3, 1}, {0, 0, 0, 1}}:t()
-- query = torch.LongTensor({0, 0, 4, 3})

actions = torch.round(torch.rand(2, 1))
yrouge = torch.rand(2, 1)

batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.1

lstm1 = build_mlp(vocab_size, embed_dim)
lstm2 = build_mlp(vocab_size, embed_dim)
lstm3 = build_mlp(vocab_size, embed_dim)

-- lstm1 = buildLSTM(vocab_size, embed_dim)
-- lstm2 = buildLSTM(vocab_size, embed_dim)
-- lstm3 = buildLSTM(vocab_size, embed_dim)

mlp1 = nn.Sequential()
mlp1:add(nn.Linear(1, embed_dim))

ParallelModel = nn.ParallelTable()
ParallelModel:add(lstm1)
ParallelModel:add(lstm2)
ParallelModel:add(lstm3)
ParallelModel:add(mlp1)

FinalMLP = nn.Sequential()
FinalMLP:add(ParallelModel)
FinalMLP:add(nn.JoinTable(2))
FinalMLP:add( nn.Linear(embed_dim * 4, 1) )
FinalMLP = FinalMLP

criterion = nn.MSECriterion()

print(sentences)
print(summary)
print(query)
print(actions)

print('sumval =', sentences[1]:sum())

for epoch=1, 100 do
    -- preds = FinalMLP:forward({query})
    preds = FinalMLP:forward({sentences, summary, query, actions})
    loss = criterion:forward(preds, yrouge)
    grads = criterion:backward(preds, yrouge)
    FinalMLP:backward({sentences, summary, query, actions}, grads)
    -- FinalMLP:backward({query}, grads)
    FinalMLP:updateParameters(learning_rate)
    FinalMLP:zeroGradParameters()
    if (epoch % 10)==0 then 
        print(string.format("Epoch %i, loss =%6f", epoch, loss))
    end
end