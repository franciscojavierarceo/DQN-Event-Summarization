
dofile("Code/utils.lua")
dofile("Code/utilsNN.lua")


genSummary = {3,1,4}
refSummary = {2,2,3}
print(genSummary, refSummary)
recall, prec, f1 = rougeScores(genSummary, refSummary)
function ngrams(dic, n)

end
print(string.format("Recall = %.3f; Precision = %.3f; F1 = %.3f", recall, prec, f1))