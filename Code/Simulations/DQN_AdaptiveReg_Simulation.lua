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

--- Stacking the models together
FullModel = nn.ConcatTable():add(LinearModel):add(LogisticModel)

print("Logistic Pred:")
pred_log = LogisticModel:forward(x)
print(pred_log)

print("Linear Pred:")
pred_reg = LinearModel:forward(x)
print(pred_reg)

print("Logistic and Linear Pred:")
pred_final = FullModel:forward(x)
print(pred_final)

nll = nn.BCECriterion()
mse = nn.MSECriterion()
pc = nn.ParallelCriterion():add(mse):add(nll)

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

actions = torch.ByteTensor(10, 2):fill(0)
maskLayer = nn.MaskedSelect()

reward = torch.rand(10):resize(10, 1)
class = torch.ones(10):resize(10, 1)

params, gradParams = FullModel:getParameters()

for i=1, 10 do
    predTotal  = FullModel:forward(x)
    predQ = predTotal[1]
    predReg = predTotal[2]

    SKIP = 1
    SELECT = 2

    for i=1, predQ:size(1) do
        if predQ[i][SKIP] > predQ[i][SELECT] then
            actions[i][SKIP] = 1
        else
            actions[i][SELECT] = 1
        end
    end
    gradParams:zero()
    predQOnActions = maskLayer:forward({predQ, actions})

    lossf = pc:forward({predQOnActions, predReg}, {reward, class})
    gradOutput = pc:backward({predQOnActions, predReg}, {reward, class})
    gradMaskLayer = maskLayer:backward({predQ, actions}, gradOutput[1])
    FullModel:backward(x, {gradMaskLayer[1], gradOutput[2]})
    FullModel:updateParameters(0.01)
    print(i, lossf)
end

print({x, gradMaskLayer, gradOutput})
print('success') 
print(FullModel:backward(x, {gradMaskLayer[1], gradOutput[2]}))