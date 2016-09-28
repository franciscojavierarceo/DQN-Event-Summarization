require 'torch'
require 'nn'
require 'rnn'
require 'csvigo'

dofile("utils.lua")

aurora_fn = '~/GitHub/DeepNLPQLearning/DO_NOT_UPLOAD_THIS_DATA/0-output/2012_aurora_shooting_first_sentence_numtext.csv'
m = csvigo.load({path = aurora_fn, mode = "large"})

mxl  = getMaxseq(m)                     --- Extracting maximum sequence length

xs = sampleData(m, 5, mxl)