require 'csvigo'

dofile("Code/utils.lua")

function buildCNN(input_path, inputfile)
    inputfile = 'cnn_data_ss.csv'

    mydata =  csvigo.load({path = input_path .. inputfile , mode = "large", verbose = false})

    print("data loaded and tokenized...")
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
            tmp[i - 1] = mydata[i][j]:split(" ")
        end

        if j == 2 then 
            queries = torch.Tensor(padZeros({tmp}, pad_l[2] ))
        end
        if j == 2 then 
            trueSummary = torch.Tensor(padZeros({tmp}, pad_l[3] ))
        end
        if j > 3 then 
            -- # of sentence tokens
            sentences[j - 3 ] = torch.Tensor(padZeros({tmp}, pad_l[j] ))
        end
    end

    print("...data exported to torch datafiles")

    torch.save(outputpath .. "cnn_data_ss.dat", {queries, trueSummary, sentences} )

end

input_path = "/home/francisco/GitHub/DQN-Event-Summarization/data/cnn_tokenized_ss/"
outputpath = "/home/francisco/GitHub/DQN-Event-Summarization/data/training_ss/"

-- input_path = "/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/data/cnn_tokenized/"
-- outputpath = "/Users/franciscojavierarceo/GitHub/DeepNLPQLearning/data/training/"

buildCNN(input_path, inputfile)

-- for i=0, 124 do 
--     buildCNN(input_path, inputfile, i)
-- end
