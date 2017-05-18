require 'csvigo'

dofile("Code/utils.lua")

function loadCNN(path, n)
    local queries, sentences, trueSummary  = {}, {}, {}
    for i = 0, 124 do
    	queries[i + 1 ] = torch.load(path .. string.format("qtokens_sid_%i.dat", i))
    	sentences[i + 1] = torch.load(path .. string.format("stokens_sid_%i.dat", i))
    	trueSummary[i + 1] = torch.load(path .. string.format("tstokens_sid_%i.dat", i))
        if n ~= nil then 
            queries[i + 1] = queries[i][{{1, n}}]
            sentences[i + 1] = sentences[i][{{1, n}}]
            trueSummary[i] = trueSummary[i][{{1, n}}]
        end
    end
    return queries, sentences, trueSummary
end

outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training/'
-- outputpath = "/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/data2/training/"
-- queries, sentences, trueSummaries = loadCNN(outputpath)

-- print(#queries[1], #sentences[1], #trueSumamries[1])
