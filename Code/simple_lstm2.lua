require 'rnn'
require 'cunn'
cuda = true

print("Running batch version of an RNN with continuous outcome")

function build_data()
    local inputs = {}
    local targets = {}
    --Use previously created and saved data
    for i = 1, dsSize do
      -- populate both tables to get ready for training
      -- local input = torch.randn(batchSize, inputSize)
        local input = torch.Tensor({{1, 2, 3, 4}, {0, 1, 1, 3}})
        local target = torch.LongTensor(batchSize):random(1, nClass)
        -- local target = torch.randn(batchSize)
        if cuda then
            input = input:float():cuda()
            target = target:float():cuda()
        end
            table.insert(inputs, input)
            table.insert(targets, target)
    end
    return inputs, targets
end

function build_network(inputSize, hiddenSize, outputSize)
    -- This works for the discrete
    model = nn.Sequential()
    :add(nn.LookupTableMaskZero(8, hiddenSize))
    :add(nn.SeqLSTM(hiddenSize, hiddenSize))
    :add(nn.Select(1, -1))                       --- Embedding layer
    :add(nn.Linear(hiddenSize, outputSize))
    :add(nn.LogSoftMax())
   -- wrap this in a Sequencer such that we can forward/backward 
   -- entire sequences of length seqLength at once
   rnn = nn.Sequencer(model)
   if cuda then
      rnn:cuda()
   end
   return rnn
end



inputSize = 6   -- Larger numbers here mean more complex problems can be solved, but can also over-fit. 256 works well for now
hiddenSize = 8 
outputSize = 2  -- We want the network to classify the inputs using a one-hot representation of the outputs
dsSize=10       -- the dataset size is the total number of examples we want to present to the LSTM 
batchSize=2     -- We present the dataset to the network in batches where batchSize << dsSize

-- And seqLength is the length of each sequence, i.e. the number of "events" we want to pass to the LSTM
-- to make up a single example. I'd like this to be dynamic ideally for the YOOCHOOSE dataset..

seqLength = 6
-- number of target classes or labels, needs to be the same as outputSize above
-- or we get the dreaded "ClassNLLCriterion.lua:46: Assertion `cur_target >= 0 && cur_target < n_classes' failed. "
nClass = 2      -- two tables to hold the *full* dataset input and target tensors

inputs, targets = build_data()
rnn = build_network(inputSize, hiddenSize, outputSize)

print('Example of inputs and outputs for a batch of data:')
print(inputs[1], targets[2])

crit = nn.ClassNLLCriterion()
seqC = nn.SequencerCriterion(crit)
if cuda then
   crit:cuda()
   seqC:cuda()
end

rnn:training()

print('Start training')
--Feed our LSTM the dsSize examples in total, broken into batchSize chunks
loss = {}
for iter=0, 100, 1 do
   local err = 0
   local start =torch.tic() 

   for offset=1, dsSize, batchSize+seqLength do
      local batchInputs = {}
      local batchTargets = {}

      -- We need to get a subset (of size batchSize) of the inputs and targets tables
      -- start needs to be "2" and end "batchSize-1" to correctly index
      -- all of the examples in the "inputs" and "targets" tables

      for i = 2, batchSize+seqLength-1,1 do
         table.insert(batchInputs, inputs[offset+i])
         table.insert(batchTargets, targets[offset+i])
      end

      if iter==0 then 
         print("This is what the seqLenght expands")
         print(batchInputs, batchTargets)
      end

      local out = rnn:forward(batchInputs)
      err = err + seqC:forward(out, batchTargets)
      gradOut = seqC:backward(out, batchTargets)
      rnn:backward(batchInputs, gradOut)

      --We update params at the end of each batch
      rnn:updateParameters(0.05)
      rnn:zeroGradParameters()
   end
   local currT = torch.toc(start)
   loss[iter] = err/dsSize
    if (iter % 10)==0 then
       print('Epoch', iter, 'loss', err/dsSize .. ' in ', currT .. ' s')
    end
end