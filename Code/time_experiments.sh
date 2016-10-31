echo "Running rmsprop on gpu"
time th dqn_model.lua --usecuda --rmsprop --nepochs 1

echo "Running rmsprop on cpu"
th dqn_model.lua --rmsprop --nepochs 1 

echo "Running sgd on gpu"
th dqn_model.lua --usecuda --nepochs 1

echo "Running sgd on cpu"
th dqn_model.lua --nepochs 1
