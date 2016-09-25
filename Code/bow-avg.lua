require 'nn'
require 'rnn'

vocab_size = 4
embeddings_dim = 3

print("**********************************************************************")
print("                                 Example 1                            ")
print("**********************************************************************")

--   Input is batch size x max sequence length. The first example has 
--   words 1, 4, and 2 and is shorter than the max sequence length so
--   we pad it with zeros.

input = torch.LongTensor{
                        {1, 4, 2, 0}, 
                        {1, 3, 3, 2}
                     }
print("Format of data in Example 1 is")
print(input)

--   Here we add the look up table which will produce output of size
--      (batch size x max sequence length x embedding dim size)

--   In this example ouptut O[1][4] is a vector of zeros because 
--   of the zero pad.

net = nn.Sequential()
net:add(nn.LookupTableMaskZero(vocab_size, embeddings_dim))

print("Dimensions of lookup table layer:")
print(#net:forward(input))
print("Lookup table layer output")
print(net:forward(input))
-- Now add the summation layer. The arguments are which dimension to
-- sum over, the total number of input dimensions, and a boolean flag
-- indicating whether or not to average the sum.

-- We are summing over the sequence which corresponds to dimension 2.
-- The input has 3 dimensions (output of lookup layer is 
-- batch x sequence x embedding) and we want to average the sum so 
-- flag is true. This is more or less what we want but there is an 
-- issue with the averaging part of the sum layer. 

-- What is it? (You can ignore it for now and throw the output of this
-- into an mlp. We can discuss how to fix it on Tuesday.)

net:add(nn.Sum(2, 3, true))
print("Dimensions after lookup table layer and averaging layer:")
print(#net:forward(input))
print("nn.sum(2,3,true) output")
print(net:forward(input))

print("**********************************************************************")
print("                                 Example 2                            ")
print("**********************************************************************")

input = torch.LongTensor{
                        {1, 4, 2, 1}, 
                        {1, 3, 3, 2}
                     }
print("Format of data in Example 2 is")
print(input)

net = nn.Sequential()
net:add(nn.LookupTableMaskZero(vocab_size, embeddings_dim))
print("Dimensions of lookup table layer:")
print(#net:forward(input))

net:add(nn.Sum(2, 3, true))

print("Dimensions after lookup table layer and averaging layer:")
print(net:forward(input))
print('Original nnsum...')
print(#net:forward(input))