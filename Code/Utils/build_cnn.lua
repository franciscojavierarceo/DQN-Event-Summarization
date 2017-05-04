require 'csvigo'
require 'math' 


dofile("Code/utils.lua")
dofile("Code/utilsNN.lua")


input_path = '/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized/'
inputfile ='cnn_data_sentence_00.csv'
mydata =  csvigo.load({path = input_path .. inputfile, mode = "large", verbose = false})

local qtokens = {}
local stokens = {}
local tstokens = {}

maxq  = 0
maxs  = 0
maxts = 0 

for i, row in pairs(mydata) do
    if i > 1 and row ~= nil then
        qtokens[i-1] = row[1]:split(" ")
        stokens[i-1] = row[2]:split(" ")
        tstokens[i-1] = row[3]:split(" ")
        maxq = math.max(maxq, #qtokens[i-1])
        maxs = math.max(maxs, #stokens[i-1])
        maxts = math.max(maxts, #tstokens[i-1])
    end
end

print(maxq, maxs, maxts)
