function policy(nnpreds, epsilon, N)
    local pred = {}
    if torch.rand(1)[1] <= epsilon then  -- Epsilon greedy strategy
        for i=1, N do
            pred[i] = (torch.rand(1)[1] > 0.5 ) and 1 or 0
        end
    else 
        --- This is the action choice 1 select, 0 skip
        for i=1, N do
            pred[i] = (nnpreds[i][1] > 0) and 1 or 0
        end
    end
    return pred
end

function repeatQuery(query_table, n)
    local out = {}
    for i=1, n do
        out[i] = query_table
    end
    return out
end

function updateTable(orig_table, insert_table, n_i)
    local out_table = {}
    for k, v in pairs(orig_table) do
        out_table[k] = v 
    end
    --- Need -1 because this starts indexing at 1
    for k, v in pairs(insert_table) do
        out_table[n_i + k - 1] = v 
    end
    return out_table
end

--- This will loop over queries
--- then iteratere over minibatches
--- then iterate over epochs

function iterateModel(batch_size, nepochs, qs, x, model, crit, epsilon, delta, mxl,
                    base_explore_rate, print_every, nuggets, learning_rate, K)
    local rscores, pscores, fscores = {}, {}, {}
    local yrouge = torch.totable(torch.randn(#x))
    local summary_list = populateOnes(#x, K)
    local action_list = torch.totable(torch.round(torch.randn(#x)))
    local preds_list = torch.totable(torch.round(torch.rand(#x)))
    for epoch=0, nepochs, 1 do
        loss = 0                    --- Compute a new MSE loss each time
        --- Reset the rougue each time
        local r_t1 , p_t1, f_t1 = 0., 0., 0.
        --- Looping over each bach of sentences for a given query
        local nbatches = torch.floor( #x / batch_size)
        for minibatch = 1, nbatches do
            if minibatch == 1 then          -- Need +1 to skip the first row
                nstart = 2
                nend = torch.round(batch_size * minibatch)
            end
            if minibatch == nbatches then 
                nstart = nend + 1
                nend = #x
            end
            if minibatch > 1 and minibatch < nbatches then 
                nstart = nend + 1
                nend = torch.round(batch_size * minibatch)
            end
            --- This step is processing the data
            local x_ss  = geti_n(x, nstart, nend)
            local xout  = grabNsamples(x_ss, 1, #x_ss)     --- Extracting N samples
            local xs  = padZeros(xout, mxl)                 --- Padding the data by the maximum length
            local qs2 = padZeros({qs}, 5)
            local qrep = repeatQuery(qs2[1], #xs)
            local preds = geti_n(preds_list, nstart, nend)
            -- Not efficient but who cares
            local sumry_ss = buildPredSummary(preds, xs, mxl)
            local sentences = torch.LongTensor(xs):t()
            print(xs[1])
            print(sumry_ss[1][1])
            -- print(sumry_ss2[1])
            print(#sumry_ss2)
            local summary = torch.LongTensor(sumry_ss2):t()
            print('pass')
            local query = torch.LongTensor(qrep):t()
            local actions = torch.Tensor(geti_n(action_list, nstart, nend)):resize(#xs, 1)
            local labels = torch.Tensor(geti_n(yrouge, nstart, nend))

            myPreds = model:forward({sentences, summary, query, actions})
            loss = loss + crit:forward(myPreds, labels)
            grads = crit:backward(myPreds, labels)
            model:backward({sentences, summary, query, actions}, grads)
            model:updateParameters(learning_rate)        -- Update parameters after each minibatch
            model:zeroGradParameters()
            preds = policy(myPreds, epsilon, #xs)
            --- Concatenating predictions into a summary
            predsummary = buildPredSummary(preds, xs)
            --- Initializing rouge metrics at time {t-1} and save scores
            for i=1, #predsummary do
                --- Calculating rouge scores; Call get_i_n() to cumulatively compute rouge
                rscores[i] = rougeRecall(geti_n(predsummary, 1, i), nuggets, K) - r_t1
                pscores[i] = rougePrecision(geti_n(predsummary, 1, i), nuggets, K) - p_t1
                fscores[i] = rougeF1(geti_n(predsummary, 1, i), nuggets, K) - f_t1
                r_t1, p_t1, f_t1 = rscores[i], pscores[i], fscores[i]
            end
            --- Updating change in rouge
            yrouge = updateTable(yrouge, fscores, nstart)
            preds_list = updateTable(preds_list, preds, nstart)
            --- Calculating last one to see actual last rouge, without delta
            rscore = rougeRecall(predsummary, nuggets, K)
            pscore = rougePrecision(predsummary, nuggets, K)
            fscore = rougeF1(predsummary, nuggets, K)
            -- print(string.format('\t Mini-batch %i/%i', minibatch, nbatches) )
            if (epoch % print_every)==0 then
                 -- This line is useful to view the min and max of the predctions
                -- if epoch > 0 then  print(myPreds:min(), myPreds:max()) end
                perf_string = string.format(
                    "Epoch %i, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                    epoch, sumTable(preds_list), #preds_list, rscore, pscore, fscore
                    )
                print(perf_string)
            end
        end
        epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
        if epsilon <= 0 then                --- leave a random exploration rate
            epsilon = base_explore_rate
        end
    end
    return model
end 