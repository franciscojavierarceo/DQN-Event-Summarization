# /bin/sh

echo "DQN BOW"
th ./dqn_model.lua --model bow  > dqn_full_bow_model_perf.txt

# These are to clear up the memory
pkill th
pkill luajit

echo "DQN LSTM"
th ./dqn_model.lua --model lstm  > dqn_full_lstm_model_perf.txt
echo "End"