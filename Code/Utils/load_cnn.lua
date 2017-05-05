require 'csvigo'

dofile("Code/utils.lua")


outputpath = '/home/francisco/GitHub/DQN-Event-Summarization/data/training/'


queries, sentences, trueSummary  = {}, {}, {}

for i = 0, 124 do
	queries[i + 1 ] = torch.load(outputpath .. string.format("qtokens_sid_%i.dat", i))
	sentences[i + 1] = torch.load(outputpath .. string.format("stokens_sid_%i.dat", i))
	trueSummary[i + 1] = torch.load(outputpath .. string.format("tstokens_sid_%i.dat", i))
end
print("Data loaded")
