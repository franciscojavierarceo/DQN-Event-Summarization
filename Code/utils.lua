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

function threshold(x, thresh)
    if x <= thresh and x > 0 then
        return 0
    else 
        return x 
    end
end
function repeatTable(input_table, n)
    local out = {}
    for i=1, n do
        out[i] = input_table
    end
    return out
end

function updateTable(orig_table, insert_table, n_i)
    local out_table = {}
    for k, v in pairs(orig_table) do
        out_table[k] = v 
    end
    --- Need -1 because this starts indexing at 1
    for i=0, #insert_table-1 do
        out_table[n_i + i] = insert_table[i+1]
    end
    return out_table
end
function makeInt(x)
    local out = {}
    for k,v in pairs(x) do
        table.insert(out, tonumber(v))
    end
    return out 
end

function buildTermDocumentTable(x, K)
    --- If K == nil then getFirstKtokens() returns x
    local out = {}
    for k,v in pairs(x) do
        out[k] = makeInt(getFirstKElements(split(x[k][1]), K))
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

function x_or_zero(pred_action, xs)
    if pred_action==1 then
        return xs
    else 
        return {0}
    end
end

function x_or_pass(pred_action, xs)
    if pred_action==1 then
        return xs
    else
        return {}
    end
end

function getFirstKElements(x, K)
    --- This function grabs the first K elements from a table
    --- e.g., getFirstKElements({1,2,3,4,5}, 3) = {1, 2, 3}
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

function getLastKElements(x, K)
    --- This function grabs the last K elements from a table
    --- e.g., getLastKElements({1,2,3,4,5}, 3) = {3,4,5}
    if K == nil then
        return x 
    end
    local out = {}
    if K > #x then 
        return x
    end
    for i=1, K, 1 do
        out[i] = x[ #x - K + i ]
    end 
    return out
end

function buildCurrentSummary(preds, xs, K)
    local out = {}
    for i = 1, #xs do
        if i == 1 then
            out[i] = {}
        elseif i == 2 then
            tmp = x_or_pass(preds[i-1], unpackZeros(xs[i-1])) 
            out[i] = getLastKElements(tmp, K)
        else 
            local tmp = tableConcat(out[i-2], x_or_pass(preds[i-1], unpackZeros(xs[i-1])))
            out[i] = getLastKElements(tmp, K)
        end 
    end
    return out 
end

function buildPredSummary(preds, xs, K)
    --- This function is used to map the token indices to extract the summary
    --- and produceds {token_id, 0, token_id} from any given *selected* sentence
    local out = {}
    for i=1, #xs do
        if i == 1 then 
            out[i] = x_or_pass(preds[i], unpackZeros(xs[i]))
        else 
            --- Update it by adding xs_i and out_{i-1}
            local tmp = tableConcat(out[i-1], x_or_pass(preds[i], unpackZeros(xs[i])))
            --- Getting the last K tokens because we want to keep last K tokens
            out[i] =  getLastKElements(tmp, K)
        end
    end
    return out
end

function Tokenize(inputdic, remove_stopwords)
    local out = {}
    --- these can be found in the total_corpus_summary.csv file
    local stopwordlist = {1, 3, 6, 23, 24, 28, 31, 54, 57, 62, 103}

    for k, v in pairs(inputdic) do
        for j, l in pairs(v) do
            if out[l] == nil then
                out[l] = 1
                else 
                out[l] = 1 + out[l]
            end
        end
    end
    if remove_stopwords then 
        for k, stopword in pairs(stopwordlist) do
            -- Removing stop words here
            if out[stopword] ~= nil then
                out[stopword] = nil
            end
        end
    end
    return out
end

--- Precision
function rougePrecision(pred_summary, ref_summaries)
    rsd = Tokenize(ref_summaries, true)
    sws = Tokenize(pred_summary, true)
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
---- Recall
function rougeRecall(pred_summary, ref_summaries)
    rsd = Tokenize(ref_summaries, true)
    sws = Tokenize(pred_summary, true)
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
function rougeF1(pred_summary, ref_summaries) 
    rnp = rougeRecall(pred_summary, ref_summaries)
    rnr = rougePrecision(pred_summary, ref_summaries)
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