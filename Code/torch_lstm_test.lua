require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

aroraname = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
wordfile = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/total_corpus_smry.csv'
nuggets = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/aurora_nuggets_numtext.csv'

m = csvigo.load({path = aroraname, mode = "large"})
w = csvigo.load({path = wordfile, mode = "large"})
q = csvigo.load({path = nuggets, mode = "large"})

N = 1000 --- Breaks at 35
K = 100
embed_dim = 3

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

function grabKtokens(x, K)
    local tmp = {}
    for k, v in pairs(x) do
        if k <= K then
            tmp[k] = v
        end
        if k > K then
            return tmp
        end
    end
    return tmp
end

function grabNsamples(x, N, K)
    local out = {}
    for k,v in pairs(m) do
        if k > 1 then
            out[k-1] = grabKtokens(split(x[k][1]), K)
        end
        if (k % (N+1))==0 then
            print(N,'elements read out of ', #x)
            break
        end
    end
    return out
end


function padZeros(x, maxlen)
    local out = {}
    for k, v in pairs(x) do
        tmp = {}
        if #v <  maxlen then
            for i=1, maxlen do
                if i <= (maxlen - #v) then
                    tmp[i] = 0
                else 
                    tmp[i] = v[i - (maxlen-#v)]
                end
            end
        else 
            tmp = v
        end
        out[k] = tmp
    end
    return out
end

--- Extracting N samples
out = grabNsamples(m, N, K)

mxl = 0
for k,v in pairs(out) do
    mxl = math.max(mxl, #v)
end

--- getting the length of the dictionary
vocab_size = 0
for k,v in pairs(out) do
    vocab_size = math.max(vocab_size, math.max(table.unpack(v)))
    if (k % N)==0 then
        print(k,'elements read out of ', #m)
        break
    end
end

--- Padding the data by the maximum length
xs = padZeros(out, mxl)

labels = torch.Tensor(torch.round(torch.rand(#out))):reshape(#out,1)

--- This is the correct format to input it
input = torch.LongTensor(xs)

LT = nn.LookupTableMaskZero(vocab_size, embed_dim)

-- For batch inputs, it's a little easier to start with sequence-length x batch-size tensor, so we transpose songData
myDataT = input:t()
batchLSTM = nn.Sequential()
batchLSTM:add(LT) -- will return a sequence-length x batch-size x embedDim tensor
batchLSTM:add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
-- print(batchLSTM:forward(myDataT)) -- sanity check
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


preds = {}
for i=1, myPreds:size()[1] do
    preds[i] = (myPreds[i][1] > 0.5) and 1 or 0 --- lua is stupid
end

print(preds)