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

function unpackZeros(x) 
    local out = {}
    local c = 1
    for k,v in pairs(x) do
        if v~=0 then
            out[c] = v
            c = c +1
        end
    end
    return out
end
function buildPredSummary(preds, xs) 
    local predsummary = {}
    c = 1
    for i=1, #xs do
        if preds[i]== 1 then
            predsummary[c] = unpackZeros(xs[i])
            c = c + 1
        end
    end
    return predsummary
end


function Tokenize(inputdic)
    local out = {}
    for k,v in pairs(inputdic) do
        for j, l in pairs(v) do
           if out[l] == nil then
                out[l] = 1
                else 
                out[l] = 1 + out[l]
            end
        end
    end
    return out
end--- Now we can calculate ROUGE
function rougeRecall(pred_summary, ref_summaries)
    rsd = Tokenize(ref_summaries)
    sws = Tokenize(pred_summary)
    for k,v in pairs(rsd) do
        if sws[k] == nil then
            sws[k] = 0
        end
    end

    for k,v in pairs(sws) do
        if rsd[k] == nil then
            rsd[k] = 0 
        end
    end
    num = 0.
    den = 0.
    for k,v in pairs(rsd) do
        num = num + math.min(rsd[k], sws[k])
        den = den + rsd[k]
    end
    return num/den
end
---- Precision
function rougePrecision(pred_summary, ref_summaries)
    rsd = Tokenize(ref_summaries)
    sws = Tokenize(pred_summary)
    for k,v in pairs(rsd) do
        if sws[k] == nil then
            sws[k] = 0
        end
    end
    for k,v in pairs(sws) do
        if rsd[k] == nil then
            rsd[k] = 0
        end
    end
    num = 0.
    den = 0.
    for k,v in pairs(rsd) do
        num = num + math.min(rsd[k], sws[k])
        den = den + sws[k]
    end
    return num/den
end
---- F1
function rougeF1(pred_summary, ref_summaries) 
    rnp = rougeRecall(pred_summary, ref_summaries)
    rnr = rougePrecision(pred_summary, ref_summaries)
    return (2. * rnp * rnr ) / (rnp + rnr)
end


function build_network(inputSize, hiddenSize, outputSize)
  model = nn.Sequential() 
  :add(nn.Linear(inputSize, hiddenSize))
  :add(nn.LSTM(hiddenSize, hiddenSize))
  :add(nn.LSTM(hiddenSize, hiddenSize))
  :add(nn.Linear(hiddenSize, outputSize))
  :add(nn.LogSoftMax())
  -- wrap this in a Sequencer such that we can forward/backward 
  -- entire sequences of length seqLength at once
  model = nn.Sequencer(rnn)
  if opt.cuda then
     model:cuda()
  end
   return model
end