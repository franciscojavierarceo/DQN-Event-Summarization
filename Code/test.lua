require 'nn'
math.randomseed(3)
torch.manualSeed(3)

-- This is the example on the documentation
-- input = {torch.rand(2,10), torch.randn(2,10)}
-- target = {torch.IntTensor{1,8}, torch.randn(2,10)}
-- nll = nn.ClassNLLCriterion()
-- mse = nn.MSECriterion()
-- pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)
-- output = pc:forward(input, target)

-- Building data
x = torch.rand(10):resize(10, 1)
xhat = x + 0.01 * torch.rand(10)
target = torch.round(x)

model1 = nn.Sequential()
model1:add(nn.Linear(1, 1))
model1:add(nn.LogSoftMax())

model2 = nn.Sequential()
model2:add(nn.Linear(1, 1))

--- Concat Table
model3 = nn.Concat(2)
model3:add(model1)
model3:add(model2)

model4 = nn.Sequential()
model4:add(nn.SelectTable(2))
-- model4:add(nn.SplitTable(2))

model5 = nn.ParallelTable()
model5:add(model4)

-- model4:add(nn.SplitTable(2))

print("Logistic Pred:")
pred_log = model1:forward(x)
print(pred_log)

print("Linear Pred:")
pred_reg = model2:forward(x)
print(pred_reg)

print("Logistic and Linear Pred:")
pred_both = model3:forward(x)
print(pred_both)

pred_final = model4:forward(x)
print(pred_final)


criterion = nn.CrossEntropyCriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(criterion):add(mse)
-- Can put in scalar if you want
-- pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)

print("NLL loss:")
nll_loss = criterion:forward(pred_log, target)
print(nll_loss)

print("MSE loss:")
mse_loss = mse:forward(pred_reg, xhat)
print(nll_loss)


print("Parallel NLL-MSE loss:")
prl_loss = pc:forward(pred_both, {target, xhat})
print(prl_loss)