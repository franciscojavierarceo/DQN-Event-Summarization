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
    if K == nil then 
        return x
    end
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
    if N == nil then 
        N = #x
    end
    for k,v in pairs(x) do
        out[k] = grabKtokens(split(x[k][1]), K)
    end
    return out
end

function getMaxseq(x)
    local maxval = 0
    for k, v in pairs(x) do
        if k > 1 then 
            seq = split(v[1])
            maxval = math.max(maxval, #seq)
        end
    end
    return maxval
end

function sampleData(x, K, mxl)
    local nbatches = torch.round( #x / K)
    for i = 1, K do
        if i == 1 then
            nstart = 2
            nend = torch.round(nbatches * i)
        end
        if i == #x then 
            nstart = nend + 1
            nend = #x
        end
        if i > 1 and i < #x then 
            nstart = nend + 1
            nend = torch.round(nbatches * i)
        end
        x_ss  = geti_n(x, nstart, nend)
        out  = grabNsamples(x_ss, 1, #x_ss)        --- Extracting N samples
        xs = padZeros(out, mxl)             --- Padding the data by the maximum length
        input = torch.LongTensor(xs)        --- This is the correct format to input it
    end
    return out
end    

function getVocabSize(x)
    local vocab_size = 0
    for k, v in pairs(x) do
        if k > 1 then
            if k== nil then k = '0' end
            seq = split(v[1])
            vocab_size = math.max(vocab_size, math.max(table.unpack(seq)) )
        end
    end
    return vocab_size
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
    for i=1, #xs do
        tmp = unpackZeros(xs[i])
        tmp1 = {0}
        if preds[i]== 1 then
            predsummary[i] = tmp
            tmp1 = tmp
        else
            predsummary[i] = tmp1
        end
    end
    return predsummary
end

function Tokenize(inputdic)
    local out = {}
    for k, v in pairs(inputdic) do
        for j, l in pairs(v) do
           if out[l] == nil then
                out[l] = 1
                else 
                out[l] = 1 + out[l]
            end
        end
    end
    return out
end
--- Necessary for the Rouge calculation for last K streams
function getLastK(x, K)
    out = {} 
    for i=0, #x-1 do
        if i <= K-1 then
            out[#x-i-1] = x[#x - i]
        else
            return out
        end
    end
    return out
end
--- Now we can calculate ROUGE
function rougeRecall(pred_summary, ref_summaries, K)
    pred_summary = getLastK(pred_summary, K)
    rsd = Tokenize(ref_summaries)
    sws = Tokenize(pred_summary)
    for k, v in pairs(rsd) do
        if sws[k] == nil then
            sws[k] = 0
        end
    end

    for k, v in pairs(sws) do
        if rsd[k] == nil then
            rsd[k] = 0 
        end
    end
    num = 0.
    den = 0.
    for k, v in pairs(rsd) do
        num = num + math.min(rsd[k], sws[k])
        den = den + rsd[k]
    end
    return num/den
end
---- Precision
function rougePrecision(pred_summary, ref_summaries, K)
    pred_summary = getLastK(pred_summary, K)
    rsd = Tokenize(ref_summaries)
    sws = Tokenize(pred_summary)
    for k, v in pairs(rsd) do
        if sws[k] == nil then
            sws[k] = 0
        end
    end
    for k, v in pairs(sws) do
        if rsd[k] == nil then
            rsd[k] = 0
        end
    end
    num = 0.
    den = 0.
    for k, v in pairs(rsd) do
        num = num + math.min(rsd[k], sws[k])
        den = den + sws[k]
    end
    return num/den
end
---- F1
function rougeF1(pred_summary, ref_summaries, K) 
    rnp = rougeRecall(pred_summary, ref_summaries, K)
    rnr = rougePrecision(pred_summary, ref_summaries, K)
    --- Had to add this line b/c f1 starts out at 0
    f1 = (rnr > 0) and (2. * rnp * rnr ) / (rnp + rnr) or 0
    return f1
end

--- Meant to cumuatively extract the elements of a table for the rouge scoring
function geti_n(x, i, n)
    local out = {}
    local c = 0
    for k,v in pairs(x) do
        if k >= i and k <= n then
            c = c + 1
            out[c] = v
        end
    end
    return out
end

function sumTable(x)
    local o = 0
    for k, v in pairs(x) do
        o = o + v
    end

    return o
end
print("...Utils file loaded")