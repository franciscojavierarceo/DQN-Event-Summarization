# /bin/sh

echo "DQN BOW"
th ./dqn_model.lua --model bow  > dqn_full_bow_model_perf.txt

# These are to clear up the memory
pkill th
pkill luajit

echo "DQN LSTM"
th ./dqn_model.lua --model lstm  > dqn_full_lstm_model_perf.txt
echo "End"

pkill th
pkill luajit

echo "DQN BOW - skip_rate = 0.25"

th ./dqn_model.lua --model bow  --skip_rate 0.25 > dqn_full_bow_model_perf_sr025.txt

pkill th
pkill luajit

echo "DQN LSTM - skip_rate = 0.25"
th ./dqn_model.lua --model lstm  --skip_rate 0.25 > dqn_full_lstm_model_perf_sr025.txt
echo "End"


pkill th
pkill luajit

echo "DQN BOW - skip_rate = 0.75"
th ./dqn_model.lua --model bow  --skip_rate 0.75 > dqn_full_bow_model_perf_sr075.txt

pkill th
pkill luajit

echo "DQN LSTM - skip_rate = 0.75"
th ./dqn_model.lua --model lstm  --skip_rate 0.75 > dqn_full_lstm_model_perf_sr075.txt
echo "End"
