require 'csvigo'

input_path = '/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized/'
inputfile ='cnn_data_sentence_00.csv'
mydata =  csvigo.load({path = input_path .. inputfile, mode = "large", verbose = false})

local qtokens = {}
local stokens = {}
local tstokens = {}

for i, row in pairs(mydata) do
	if i > 1 and row ~= nil then
		qtokens[i-1] = row[1]:split(" ")
		stokens[i-1] = row[2]:split(" ")
		tstokens[i-1] = row[3]:split(" ")
		--qtokens[i-1] = row[1]
		--stokens[i-1] = row[2]
		--tstokens[i-1] = row[3]
	end
end

print(stokens[1])
