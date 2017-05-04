require 'csvigo'
require 'math' 

input_path = '/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized/'
inputfile ='cnn_data_sentence_00.csv'
mydata =  csvigo.load({path = input_path .. inputfile, mode = "large", verbose = false})

local qtokens = {}
local stokens = {}
local tstokens = {}


function makeInt(x)
    --- Casts string values into integers when reading in the data
    local out = {}
    for k,v in pairs(x) do
        table.insert(out, tonumber(v))
    end
    return out 
end


maxq = 0
maxs = 0
maxts = 0

for i, row in pairs(mydata) do
	if i > 1 and row ~= nil then
		qtokens[i-1]  = makeInt(row[1]:split(" "))
		stokens[i-1]  = makeInt(row[2]:split(" "))
		tstokens[i-1] = makeInt(row[3]:split(" "))
		maxq = math.max(maxq, #qtokens[i-1])
		maxs = math.max(maxs, #stokens[i-1])
		maxts = math.max(maxts, #tstokens[i-1])
	end
end

print(stokens[{{1,4}}])
print(maxq, maxs, maxts)
