require 'os'
require 'nn'
require 'rnn'
require 'optim'
require 'parallel'

function genNbyK(n, k, a, b)
    out = torch.LongTensor(n, k)
    for i=1, n do
        for j = 1, k do
            out[i][j] = torch.random(a, b)
        end
    end
    return out
end

function buildTokenCounts(summary, stopwordlist)
    local counts = {}
    if #summary:size() == 1 then
        summary = summary:resize(1, summary:size(1))
    end
    for i=1, summary:size(1) do
        for j=1, summary:size(2) do
            if summary[i][j] > 0 then
                local token = summary[i][j]
                if counts[token] == nil then
                    counts[token] = 1
                else
                    counts[token] = counts[token] + 1
                end
            end
        end
    end
    return counts
end
function rougeScores(genSummary, refSummary)
    local genTotal = 0
    local refTotal = 0
    local intersection = 0
    -- Inserting the missing keys
    for k, genCount in pairs(genSummary) do
        if refSummary[k] == nil then
            refSummary[k] = 0
        end
    end
    for k, refCount in pairs(refSummary) do
        local genCount = genSummary[k]
        if genCount == nil then 
            genCount = 0 
        end
        intersection = intersection + math.min(refCount, genCount)
        refTotal = refTotal + refCount
        genTotal = genTotal + genCount
    end
    recall = intersection / refTotal
    prec = intersection / genTotal
    if refTotal == 0 then
        recall = 0
    end 
    if genTotal == 0 then
        prec = 0
    end
    -- tmp = {intersection, refTotal, genTotal}
    if recall > 0 or prec > 0 then
        f1 = (2 * recall * prec) / (recall + prec)
    else 
        f1 = 0
    end
    -- return recall, prec, f1
    return f1
end

function worker()
    require 'sys'
    require 'torch'
    require 'math'

    function buildTokenCounts(summary, stopwordlist)
        local counts = {}
        if #summary:size() == 1 then
            summary = summary:resize(1, summary:size(1))
        end
        for i=1, summary:size(1) do
            for j=1, summary:size(2) do
                if summary[i][j] > 0 then
                    local token = summary[i][j]
                    if counts[token] == nil then
                        counts[token] = 1
                    else
                        counts[token] = counts[token] + 1
                    end
                end
            end
        end
        return counts
    end
    function rougeScores(genSummary, refSummary)
        local genTotal = 0
        local refTotal = 0
        local intersection = 0
        -- Inserting the missing keys
        for k, genCount in pairs(genSummary) do
            if refSummary[k] == nil then
                refSummary[k] = 0
            end
        end
        for k, refCount in pairs(refSummary) do
            local genCount = genSummary[k]
            if genCount == nil then 
                genCount = 0 
            end
            intersection = intersection + math.min(refCount, genCount)
            refTotal = refTotal + refCount
            genTotal = genTotal + genCount
        end

        recall = intersection / refTotal
        prec = intersection / genTotal
        if refTotal == 0 then
            recall = 0
        end 
        if genTotal == 0 then
            prec = 0
        end
        if recall > 0 or prec > 0 then
            f1 = (2 * recall * prec) / (recall + prec)
        else 
            f1 = 0
        end
        return recall, prec, f1
    end

    while true do
        m = parallel.yield() -- yield = allow parent to terminate me
        if m == 'break' then 
            break 
        end
        input = parallel.parent:receive()  -- receive data
        nforks = input.data[3]
        chunksize = math.floor(input.data[1]:size(1) / nforks)
        start_index = chunksize * (parallel.id - 1) + 1
        end_index = chunksize * parallel.id
        -- Prints the indices and shows that it's working
        data_ss = input.data[1][parallel.id]
        true_ss = input.data[2][parallel.id]
        -- print(parallel.id, start_index, end_index, nforks)
        -- data_ss = input.data[1][{{start_index, end_index}}]
        -- true_ss = input.data[2][{{start_index, end_index}}]

        nd = data_ss:size(1)
        perf = torch.zeros(nd)
        -- print(start_index, chunksize)
        for i=start_index, chunksize do
            perftmp = rougeScores(
                        buildTokenCounts(data_ss[i]), 
                        buildTokenCounts(true_ss[i])
                    )
            perf[i] = perftmp
        end

        -- parallel.parent:send(data_ss:size(1))
        parallel.parent:send(perf)
    end
end

function parent(input)
    parallel.nfork(input[3])
    parallel.children:exec(worker)
    send_var = {name='my variable', data=input} 

    parallel.children:join()
    parallel.children:send(send_var)
    replies = parallel.children:receive()
    parallel.children:join('break')
    -- Returns the data in order of the index
    -- print(replies)
    -- print(torch.Tensor(replies))
    return replies
end

n = 16
nforks = 16
k = 5
a = 1
b = 20

xs = genNbyK(n, k, a, b)
truexs = genNbyK(n, k, a, b)

nClock = os.clock()
ok, scores_p = pcall(parent, {xs, truexs, nforks})
-- print(scores_p)
ptime = os.clock()-nClock

if not ok then 
    print(err) 
end

parallel.close()

nClock = os.clock()
scores_l = torch.zeros(n)
for i=1, n do
    perf = rougeScores(
                buildTokenCounts(xs[i]), 
                buildTokenCounts(truexs[i])
            )
    scores_l[i] = perf
end

-- print(scores_p)
-- print(scores_l:totable())

out = (torch.Tensor(scores_p) - torch.Tensor(scores_l) ):sum() 
ltime = os.clock()-nClock

print(string.format("Delta = %5.f" %  out ))
print(string.format("Parallel Elapsed time: %.5f" %  ptime) )
print(string.format("Linear Elapsed time: %.5f" % ltime ))
print(string.format("Parallel/Linear Elapsed time: %.5f" % (ptime/ltime) ))

