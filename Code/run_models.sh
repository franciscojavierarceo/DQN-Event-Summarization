# /bin/sh

# echo "Random model"
# th ./dqn_random.lua

echo "DQN BOW"
th ./run_model_all.lua --model bow  > dqn_full_bow_model_perf.txt

echo "DQN LSTM"
th ./run_model_all.lua --model lstm  > dqn_full_lstm_model_perf.txt
echo "End"