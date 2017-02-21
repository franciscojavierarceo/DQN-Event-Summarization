require 'os'
require 'nn'
require 'rnn'
require 'optim'
require 'parallel'

function worker()
    require 'sys'
    require 'torch'
    while true do
        m = parallel.yield() -- yield = allow parent to terminate me
        if m == 'break' then 
            break 
        end
        local t = parallel.parent:receive()  -- receive data
        print(parallel.id)
        parallel.parent:send(t.data)

    end
end


function parent(input)
    parallel.nfork(10)
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
