require 'nn'
math.randomseed(3)
torch.manualSeed(3)


-- Building data
x = torch.rand(10):resize(10, 1)
xhat = x + 0.01 * torch.rand(10)
target = torch.round(x)


model1 = nn.Sequential()
model1:add(nn.Linear(1, 1))
model1:add(nn.LogSoftMax())

model2 = nn.Sequential()
model2:add(nn.Linear(1, 1))

model3 = nn.Parallel(1, 1)
model3:add(model1)
model3:add(model2)

print("Logistic Pred:")
print(model1:forward(x))

print("Linear Pred:")
print(model2:forward(x))

print("Logistic and Linear Pred:")
-- print(model3:forward(x))

criterion = nn.CrossEntropyCriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(criterion):add(mse)
-- Can put in scalar if you want
-- pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)

print("NLL loss:")
pred_log = model1:forward(x)
nll_loss = criterion:forward(pred_log, target)
print(nll_loss)

print("MSE loss:")
pred_reg = model2:forward(x)
mse_loss = mse:forward(pred_reg, xhat)
print(nll_loss)


print("Parallel NLL-MSE loss:")
pred_both = model3:forward(x)
prl_loss = pc:forward(pred_both, {target, xhat})
print(prl_loss)

