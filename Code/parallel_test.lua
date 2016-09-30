require 'rnn'

-- mlp = nn.Sequential()       -- Create a network that takes a Tensor as input
-- mlp:add(nn.SplitTable(2))
-- c = nn.ParallelTable()      -- The two Tensor slices go through two different Linear
-- c:add(nn.Linear(10, 3))     -- Layers in Parallel
-- c:add(nn.Linear(5, 7))
-- mlp:add(c)                  -- Outputing a table with 2 elements
-- p = nn.ParallelTable()      -- These tables go through two more linear layers separately
-- p:add(nn.Linear(3, 2))
-- p:add(nn.Linear(7, 1))
-- mlp:add(p)
-- mlp:add(nn.JoinTable(1))    -- Finally, the tables are joined together and output.


xs = torch.LongTensor{{0, 1, 3, 4}, {2, 1, 4, 3}}
x = xs:t()
y = torch.randn(10)
z = torch.rand(5)

vocab_size = 4
embed_dim = 3
outputSize = 1

lstm = nn.Sequential()
lstm:add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
lstm:add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
lstm:add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
lstm:add(nn.SelectTable(-1)) -- selects last state of the LSTM
lstm:add(nn.Linear(embed_dim, outputSize)) -- map last state to a score for classification

p = nn.ParallelTable()
p:add(nn.Linear(10, 2))
p:add(nn.Linear(5, 3))
-- lstm:add(p)

print(lstm:forward(x))
print(p:forward{y, z})

lstm2 = nn.Sequential()
lstm2:add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
lstm2:add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
lstm2:add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
lstm2:add(nn.SelectTable(-1)) -- selects last state of the LSTM
lstm2:add(nn.Linear(embed_dim, outputSize)) -- map last state to a score for classification

g = nn.ParallelTable()
g:add(nn.Linear(10, 2))
g:add(nn.Linear(5, 3))
lstm2:add(g)

-- print(lstm2:forward{x,y,z})
-- print(lstm2:forward(x,y,z))
mlp = nn.Sequential()         -- Create a network that takes a Tensor as input
c = nn.ConcatTable()          -- The same Tensor goes through two different Linear
c:add(nn.Linear(10, 3))       -- Layers in Parallel
c:add(nn.Linear(10, 7))
mlp:add(c)                    -- Outputing a table with 2 elements
p = nn.ParallelTable()        -- These tables go through two more linear layers
p:add(nn.Linear(3, 2))        -- separately.
p:add(nn.Linear(7, 1))
mlp:add(p)
mlp:add(nn.JoinTable(1))      -- Finally, the tables are joined together and output.

print(torch.randn(10))
pred = mlp:forward(torch.randn(10))
print(pred)

-- pred = lstm:forward(x, y, z)
-- pred = lstm:forward{x, y, z}

-- y:copy(x:select(2, 1):narrow(1, 1, 3))
-- pred = mlp:forward(x)

-- criterion = nn.MSECriterion()
-- err = criterion:forward(pred, y)
-- gradCriterion = criterion:backward(pred, y)
-- mlp:zeroGradParameters()
-- mlp:backward(x, gradCriterion)
-- mlp:updateParameters(0.05)
