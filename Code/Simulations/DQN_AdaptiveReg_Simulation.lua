require 'nn'
math.randomseed(3)
torch.manualSeed(3)
-- Building data
x = torch.randn(20):resize(10, 2)
xhat = x + 0.01 * torch.rand(20):resize(10, 2)
b = torch.rand(2, 1)
yhat = torch.mm(x, b)  + torch.randn(10)  -- np.dot()
n = yhat:size(1)
onex = torch.ones(n):resize(n, 1)
target = torch.round(nn.Sigmoid():forward(yhat))
print(target)
-- This is the Logistic
LogisticModel = nn.Sequential():add(nn.Linear(2, 1)):add(nn.LogSoftMax())

-- This is the linear regression
LinearModel = nn.Sequential():add(nn.Linear(2, 2))

--- Concat Table
-- model3 = nn.Parallel(2,2):add(model1):add(model2)
FullModel = nn.ConcatTable():add(LinearModel):add(LogisticModel)
-- model3 = nn.Concat(2):add(model1):add(model2)

-- model4 = nn.Sequential()
-- model4:add(model3)
-- model4:add(nn.SplitTable(2))

print("Logistic Pred:")
pred_log = LogisticModel:forward(x)
print(pred_log)

print("Linear Pred:")
pred_reg = LinearModel:forward(x)
print(pred_reg)

print("Logistic and Linear Pred:")
pred_final = FullModel:forward(x)
print(pred_final)

-- print("Joined and separated")
-- pred_final = model4:forward(x)
-- print(pred_final)

nll = nn.BCECriterion()
-- nll = nn.ClassNLLCriterion()
-- nll = nn.CrossEntropyCriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(mse):add(nll)
-- Can put in scalar if you want
-- pc = nn.ParallelCriterion():add(nll, 0.5):add(mse)

print("NLL loss:")
nll_loss = nll:forward(pred_log, target)
print(nll_loss)

print("MSE loss:")
mse_loss = mse:forward(pred_reg, xhat)
print(nll_loss)

print("Parallel NLL-MSE loss:")
prl_loss = pc:forward( {pred_reg, pred_log}, {xhat, target} )
print(prl_loss)

print("Parallel Full NLL-MSE loss:")
prl_final = pc:forward(pred_final, {xhat, target})
print(prl_final)