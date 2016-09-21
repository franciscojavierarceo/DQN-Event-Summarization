require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

filename = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_numtext.csv'

m = csvigo.load({path = filename, mode = "large"})

N = 35 --- Breaks at 35
K = 5
embed_dim = 6


function split(pString)
   local pPattern = " "
   local Table = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pPattern
   local last_end = 1
   local s, e, cap = pString:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(Table, cap)
      end
      last_end = e + 1
      s, e, cap = pString:find(fpat, last_end)
   end
   if last_end <= #pString then
      cap = pString:sub(last_end)
      table.insert(Table, cap)
   end
   return Table
end

function grabn(x, n)
    local tmp = {}
    for k, v in pairs(x) do
        if k <= n then
            tmp[k] = tonumber(v)
        end
        if k > n then
            return tmp
        end
    end
end

--- putting data into table
out = {}
for k,v in pairs(m) do
    if k > 1 then
        out[k-1] = split(m[k][1])
        --- out[k-1] = grabn(split(m[k][1]), K)
    end
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
end

--- getting the length of the dictionary/ vocab_size of our data
vocab_size = 0
for k,v in pairs(out) do
    vocab_size = math.max(vocab_size, math.max(table.unpack(v)))
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
end

labels = torch.Tensor(torch.round(torch.rand(#out))):reshape(#out,1)

print(#labels, labels:sum(), labels:mean())

input = torch.LongTensor(out)   --- This is the correct format to input it, not {}

-- LT = nn.LookupTable(vocab_size, embed_dim)
LT = nn.LookupTableMaskZero(vocab_size, embed_dim)
-- For batch inputs, it's a little easier to start with sequence-length x batch-size tensor, so we transpose songData
myDataT = input:t()
batchLSTM = nn.Sequential()
batchLSTM:add(LT) -- will return a sequence-length x batch-size x embedDim tensor
batchLSTM:add(nn.SplitTable(1, 3)) -- splits into a sequence-length table with batch-size x embedDim entries
--- print(batchLSTM:forward(myDataT)) -- sanity check
-- now let's add the LSTM stuff
batchLSTM:add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
batchLSTM:add(nn.SelectTable(-1)) -- selects last state of the LSTM
batchLSTM:add(nn.Linear(embed_dim, 1)) -- map last state to a score for classification
batchLSTM:add(nn.Sigmoid()) -- convert score to a probability
myPreds = batchLSTM:forward(myDataT)
print(#myPreds)

-- we can now call :backward() as follows
bceCrit = nn.BCECriterion()
loss = bceCrit:forward(myPreds, labels)
dLdPreds = bceCrit:backward(myPreds, labels)
batchLSTM:backward(myDataT, dLdPreds)

print(loss)