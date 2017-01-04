
dofile("Code/utils.lua")
dofile("Code/utilsNN.lua")


genSummary = {3, 1, 4, 3, 1}
refSummary = {2, 2, 3, 3, 1}
-- print(genSummary, refSummary)
recall, prec, f1 = rougeScores(genSummary, refSummary)

function bigrams(dic)
    local out = {}
    for i=1, #dic do 
        for j=1, #dic do
            if i == j then
                indx = string.format("%i", dic[i])
                if out[indx] == nil then
                    out[indx] = 1
                else 
                    out[indx]  = out[indx] + 1
                end
            else
                indx = string.format("%i, %i", dic[i], dic[j])
                if out[indx] == nil then
                    out[indx] = 1
                else 
                    out[indx]  = out[indx] + 1
                end
            end
        end
    end
    return out
end
print(ngrams(genSummary))
print(string.format("Recall = %.3f; Precision = %.3f; F1 = %.3f", recall, prec, f1))