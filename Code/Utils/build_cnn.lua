require 'csvigo'

dofile("Code/utils.lua")

function buildCNN(input_path, inputfile, idx)
    inputfile = 'cnn_data_ss.csv'

    mydata =  csvigo.load({path = input_path .. inputfile , mode = "large", verbose = false})

    -- Initialize padding length
    pad_l = {}
    for i = 1, #mydata[1] do 
        pad_l[i] = 0
    end

    for i=1, #mydata do 
        for j=1, #pad_l do 
            pad_l[j] = math.max( pad_l[j],  #mydata[i][j]:split(" "))
        end
    end
    sentences = {}
    for j = 1, 125 do 
        tmp = {}
        for i = 2, #mydata do 
            tmp[i] = mydata[i][j + 3]:split(" ")
        end
        -- # of sentence tokens
        sentences[j] = torch.Tensor(padZeros({tmp}, pad_l[j + 3] ))
    end

    torch.Tensor(padZeros(qtokens))
    local qtokens, stokens, tstokens = {}, {}, {}
    local maxq, maxs, maxts = 0, 0, 0

    for i, row in pairs(mydata) do
        if i > 1 and row ~= nil then
            qtokens[i-1] = row[3]:split(" ")
            stokens[i-1] = row[4]:split(" ")
            tstokens[i-1] = row[5]:split(" ")
            maxq = math.max(maxq, #qtokens[i-1])
            maxs = math.max(maxs, #stokens[i-1])
            maxts = math.max(maxts, #tstokens[i-1])
        end
    end
    print(string.format("data %i loaded and tokenized...", idx))

    local q_x = torch.Tensor(padZeros(qtokens, maxq))
    local s_x = torch.Tensor(padZeros(stokens, maxs))
    local ts_x = torch.Tensor(padZeros(tstokens, maxts))

    torch.save(outputpath .. string.format("qtokens_sid_%i.dat", idx), q_x)
    torch.save(outputpath .. string.format("stokens_sid_%i.dat", idx), s_x)
    torch.save(outputpath .. string.format("tstokens_sid_%i.dat", idx), ts_x)

    print("...data exported to torch datafiles")
end

input_path = "/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized_ss/"
outputpath = "/home/francisco/GitHub/DQN-Event-Summarization/data/training_ss/"

-- input_path = "/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/data/cnn_tokenized/"
-- outputpath = "/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/data/training/"

for i=0, 124 do 
    buildCNN(input_path, inputfile, i)
end
