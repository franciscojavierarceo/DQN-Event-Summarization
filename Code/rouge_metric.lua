---- Note: Here I tokenize the data differently than the string version
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

--- example word tokens
ref_texts0 = {{1}}                  --- Single Reference Document
ref_texts1 = {{1, 2}}               --- Single Reference Document with 2 words
ref_texts2 = {{1}, {1,2}}           --- Two Reference Documents with varying words
ref_texts3 = {{1}, {1,2}, {3,2}}    --- Three Reference documents with varying words
summary_pred = {3, 1}

print("Rouge Recall scores are")
for k,v in pairs({ref_texts0, ref_texts1, ref_texts2, ref_texts3}) do
    print(rougeRecall({summary_pred},v))
end

print("Rouge Precision scores are")
for k,v in pairs({ref_texts0, ref_texts1, ref_texts2, ref_texts3}) do
    print(rougePrecision({summary_pred},v))
end

print("Rouge F1 scores are")
for k,v in pairs({ref_texts0, ref_texts1, ref_texts2, ref_texts3}) do
    print(rougeF1({summary_pred},v))
end