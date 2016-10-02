require 'nn'
require 'rnn'

function buildLSTM(vsize, edim, odim)
    local lstm = nn.Sequential()
    lstm:add(nn.LookupTableMaskZero(vsize, edim))
    lstm:add(nn.SplitTable(1, edim))
    lstm:add(nn.Sequencer(nn.LSTM(edim, edim)))
    lstm:add(nn.SelectTable(-1))
    return lstm
end

sentences = torch.LongTensor{{0, 1, 3, 4}, {2, 1, 4, 3}}:t()
summary = torch.LongTensor{{0, 0, 1, 4}, {0, 2, 3, 1}}:t()
query = torch.LongTensor{{0, 0, 4, 3}, {0, 1, 3, 2}}:t()
actions = torch.round(torch.rand(2, 1))
yrouge = torch.rand(2, 1)

batch_size = 2
vocab_size = 4
embed_dim = 10
outputSize = 1
learning_rate = 0.1

lstm1 = buildLSTM(vocab_size, embed_dim, outputSize)
lstm2 = buildLSTM(vocab_size, embed_dim, outputSize)
lstm3 = buildLSTM(vocab_size, embed_dim, outputSize)

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

criterion = nn.MSECriterion()

for epoch=1, 100 do
    preds = FinalMLP:forward({sentences, summary, query, actions})
    loss = criterion:forward(preds, yrouge)
    grads = criterion:backward(preds, yrouge)
    FinalMLP:backward({sentences, summary, query, actions}, grads)
    FinalMLP:updateParameters(learning_rate)
    FinalMLP:zeroGradParameters()
    print(string.format("Epoch %i, loss =%6f", epoch, loss))
end