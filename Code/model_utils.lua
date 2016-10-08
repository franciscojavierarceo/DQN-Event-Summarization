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
    for i=0, #insert_table-1 do
        out_table[n_i + i] = insert_table[i+1]
    end
    return out_table
end


function build_bowmlp(vocab_size, embed_dim)
    local model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, embed_dim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    -- :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse

   return model
end

function build_lstm(vocab_size, embed_dim)
    local model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    -- :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_model(model, vocab_size, embed_dim, outputSize, use_cuda)
    if model == 'bow' then
        print("Running BOW model")
        mod1 = build_bowmlp(vocab_size, embed_dim)
        mod2 = build_bowmlp(vocab_size, embed_dim)
        mod3 = build_bowmlp(vocab_size, embed_dim)
    end
    if model == 'lstm' then         
        print("Running LSTM model")
        mod1 = build_lstm(vocab_size, embed_dim)
        mod2 = build_lstm(vocab_size, embed_dim)
        mod3 = build_lstm(vocab_size, embed_dim)
    end

    local mod4 = nn.Sequential()
    mod4:add(nn.Linear(1, embed_dim))
    -- mlp1:add(nn.ReLU())

    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)
    ParallelModel:add(mod4)

    local FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add( nn.Linear(embed_dim * 4, outputSize) )

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end

--- This will loop over queries
--- then iteratere over minibatches
--- then iterate over epochs

function iterateModel(batch_size, nepochs, qs, x, sent_file, model, crit, epsilon, delta, mxl,
                    base_explore_rate, print_every, nuggets, learning_rate, K, use_cuda)
    if use_cuda then
      Tensor = torch.CudaTensor
      LongTensor = torch.CudaLongTensor
      crit = crit:cuda()
      print("...running on GPU")
    else
      Tensor = torch.Tensor
      LongTensor = torch.LongTensor
      print("...running on CPU")
    end
    local yrouge = torch.totable(torch.randn(#x))
    local action_list = torch.totable(torch.round(torch.rand(#x)))
    local preds_list = torch.totable(torch.round(torch.rand(#x)))
    preds_list[1] = 0
    local ss_list = grabNsamples(sent_file, 1, #sent_file)

    print_string = string.format(
        "training model with learning rate = %.3f, K = %i, and minibatch size = %i...",
                learning_rate, K, batch_size)
    print(print_string)

    for epoch=0, nepochs, 1 do
        loss = 0.                    --- Compute a new MSE loss each time
        --- Reset the rougue each epoch
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
            -- Find the optimal actions / predictions
            -- local preds = geti_n(preds_list, nstart, nend)
            --- Update the summary every mini-batch
            local sumry_list = buildKSummary(preds_list, ss_list, K)
            --- Rebuilding entire prediction each time
            local sumry_ss = geti_n(sumry_list, nstart, nend)

            local summary = LongTensor(sumry_ss):t()
            local sentences = LongTensor(xs):t()
            local query = LongTensor(qrep):t()
            local actions = Tensor(geti_n(preds_list, nstart, nend)):resize(#xs, 1)

            if use_cuda then
                 actions =  actions:cuda()
            end

            --- Run forward pass to evaluate our data 
            myPreds = model:forward({sentences, summary, query, actions})
            -- print(geti_n(torch.totable(myPreds), 1 , 5) )

            if use_cuda then
                myPreds = myPreds:double()
            end

            --- Execute policy based on our E[ROUGUE]
                --- Notice that myPreds gives us our action by returning
                    ---  select if E[ROUGUE] > 0 or skip if E[ROUGUE] <= 0
            preds = policy(myPreds, epsilon, #xs)
            --- Concatenating predictions into a summary
            predsummary = buildPredSummary(preds, xs, K)

            --- Initializing rouge metrics at time {t-1} and save scores
            local rscores, pscores, fscores = {}, {}, {}
            -- Now we evaluate our action through the critic/Oracle
            for i=1, #predsummary do
                --- Calculating rouge scores; Call get_i_n() to cumulatively compute rouge
                rscores[i] = rougeRecall(buildPredSummary(geti_n(preds, 1, i), geti_n(xout, 1, i)), nuggets) - r_t1
                pscores[i] = rougePrecision(buildPredSummary(geti_n(preds, 1, i), geti_n(xout, 1, i)), nuggets) - p_t1
                fscores[i] = rougeF1(buildPredSummary(geti_n(preds, 1, i), geti_n(xout, 1, i)), nuggets) - f_t1
                r_t1, p_t1, f_t1 = rscores[i], pscores[i], fscores[i]
            end

            local labels = torch.Tensor(pscores)
            if use_cuda then
                 labels = labels:cuda()
                 myPreds = myPreds:cuda()
            end

            -- Now we backpropagate our observed outcomes
            loss = loss + crit:forward(myPreds, labels)
            grads = crit:backward(myPreds, labels)
            model:zeroGradParameters()
            model:backward({sentences, summary, query, actions}, grads)
            model:updateParameters(learning_rate)        

            -- Updating our bookkeeping arrays
            yrouge = updateTable(yrouge, pscores, nstart)
            preds_list = updateTable(preds_list, preds, nstart)
            predsummary2 = buildPredSummary(preds_list, xs, K)

            --- Calculating last one to see actual last rouge, without delta
            rscore = rougeRecall(predsummary2, nuggets, K)
            pscore = rougePrecision(predsummary2, nuggets, K)
            fscore = rougeF1(predsummary2, nuggets, K)

            if (epoch % print_every)==0 then
                perf_string = string.format(
                    "Epoch %i, epsilon = %.3f, minibatch %i/%i, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                    epoch, epsilon, minibatch, nbatches, sumTable(preds), #preds, rscore, pscore, fscore
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