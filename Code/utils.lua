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

function populateOnes(n, ml)
    local out = {}
    local tmp = {}
    for j=1, ml do
        tmp[j] = 1
    end
    for i=1, n do
        out[i] = tmp
    end
    return out
end

function populateZeros(n, ml)
    local out = {}
    local tmp = {}
    for j=1, ml do
        tmp[j] = 0
    end
    for i=1, n do
        out[i] = tmp
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

function tableConcat(input_t1, input_t2)
    local out_table = {}
    c = 1 
    for k,v in pairs(input_t1) do
        out_table[c] = v
        c = c + 1
    end 
    for k,v in pairs(input_t2) do
        out_table[c] = v
        c = c + 1
    end
    return out_table
end

function zero_or_x(pred_action, xs)
    if pred_action==1 then
        return xs
    else 
        return {0}
    end
end

function buildSummary(preds, xs, maxlen)
    local out = {}
    for i=1, #xs do
        if i == 1 then 
            out[i] = zero_or_x(preds[i], unpackZeros(xs[i]))
        else 
            --- Update it by adding xs_i and out_{i-1}
            out[i] = tableConcat(zero_or_x(preds[i], unpackZeros(xs[i])), out[i-1])
        end
    end
    if maxlen == 0 then
        maxlen = 0
        for k,v in pairs(out) do
            maxlen = math.max(maxlen, #v)
        end
    end
    outpadded = padZeros(out, maxlen)
    return outpadded
end

function initializePredSummary(pred_action, xs, mxl) 
    local predsummary = {}
    --- TODO:
        --- NEED TO FIX THIS TO KEEP THE LAST K SENTENCES AND CONCATENATE THEM
        --- INTO A LOGICAL SEQUENCE
    local tmp1 = {}
    for i=1, #xs do
        tmp = padZeros(unpackZeros(xs[i]), mxl)
        if pred_action[i]== 1 then
            table.insert(tmp1, tmp)
            predsummary[i] = tmp1
        else
            predsummary[i] = tmp1
        end
    end
    return predsummary
end
function buildPredSummary(pred_action, xs) 
    local predsummary = {}
    --- This looks stupid but it's right because we have to retain
    --- the tmp1 when it's not 1, so it's a running total
    local tmp1 = {}
    for i=1, #xs do
        tmp = unpackZeros(xs[i])
        if pred_action[i]== 1 then
            table.insert(tmp1, tmp)
            predsummary[i] = tmp1
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
    --- Note: need to use K-1 bc we start at 0
    if K == nil then
        return x
    end
    out = {}
    k = 0
    for i=0, #x-1 do
        if sumTable(x[#x - i])  ~= 0 then
            if k <= K-1 then
                --- 10 - 0 - 1 = 9..., 8, 7 ..., 1
                out[#x-i-1] = x[#x - i]
                k = k + 1
            else
                return out
            end
        end
    end
    return out
end
--- Now we can calculate ROUGE
function rougeRecall(pred_summary, ref_summaries, K)
    -- pred_summary = getLastK(pred_summary, K)
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
    return (den > 0) and num/den or 0
end
---- Precision
function rougePrecision(pred_summary, ref_summaries, K)
    -- pred_summary = getLastK(pred_summary, K)
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
    return (den > 0) and num/den or 0
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
    local c = 1
    for k,v in pairs(x) do
        if k >= i and k <= n then
            out[c] = v
            c = c + 1
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