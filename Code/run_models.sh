# /bin/sh

echo "Random model"
th ./dqn_random.lua

echo "DQN BOW"
th ./dqn_bow_mlp.lua --model bow

echo "DQN LSTM"
th ./dqn_lstm_mlp.lua --model lstm

echo "End"