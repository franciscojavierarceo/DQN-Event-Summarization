require 'csvigo'

dofile("Code/utils.lua")

function loadCNN(path)
    local queries, sentences, trueSummary  = {}, {}, {}
    for i = 0, 124 do
    	queries[i + 1 ] = torch.load(path .. string.format("qtokens_sid_%i.dat", i))
    	sentences[i + 1] = torch.load(path .. string.format("stokens_sid_%i.dat", i))
    	trueSummary[i + 1] = torch.load(path .. string.format("tstokens_sid_%i.dat", i))
    end

    print("Data loaded")
    return queries, sentences, trueSummary
end

outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training/'
mydata = loadCNN(outputpath)

print(#mydata[1], #sentences, #trueSummary)