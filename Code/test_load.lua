require 'torch'
require 'rnn'
local dl = require 'dataload'

dataloader = dl.DataLoader()

indices = torch.LongTensor{1,2,3,4,5}
inputs, targets = dataloader:index(indices)