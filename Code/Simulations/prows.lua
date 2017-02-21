require 'os'
require 'nn'
require 'rnn'
require 'optim'
require 'parallel'


nforks = 10
function worker()
    require 'sys'
    require 'torch'
    require 'math'
    nforks = 10
    while true do
        m = parallel.yield() -- yield = allow parent to terminate me
        if m == 'break' then 
            break 
        end
        local t = parallel.parent:receive()  -- receive data
        -- print(math.floor(4/10))
        -- print(t.data:size(1) / nforks, parallel.id, t.data:size(1))
        chunksize = math.floor(t.data:size(1) / nforks)
        start_index = chunksize * (parallel.id-1) + 1
        end_index = chunksize * parallel.id
        -- print(parallel.id, start_index, end_index, t.data[{start_index, end_index}])
        print(parallel.id, start_index, end_index, t.data[{{start_index, end_index}}]:size(1))
        -- xs[{{from, to}}]
        parallel.parent:send(t.data)
    end
end

function parent(input)
    parallel.nfork(nforks)
    parallel.children:exec(worker)
    send_var = {name='my variable', data=input} 

    parallel.children:join()
    parallel.children:send(send_var)
    replies = parallel.children:receive()
    parallel.children:join('break')
    print(replies)
end


xs = torch.randn(100, 10)
xs2 = torch.randn(100, 10)
-- protected execution
ok, err = pcall(parent, xs)
if not ok then 
    print(err) 
end

parallel.close()
