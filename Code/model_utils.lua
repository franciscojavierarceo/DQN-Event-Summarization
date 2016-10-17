function policy(nnpreds, epsilon)
    --- This executes our policy over our predicted rougue from the NN
    local output = {}
    local N = #nnpreds
    -- Epsilon greedy strategy
    if torch.rand(1)[1] <= epsilon then  
        output = torch.totable(torch.rand(N,2))
    else     --- This is the action choice 1 select, 0 skip
        output =  nnpreds
    end
    return output
end

function policy2(nnpreds, epsilon)
    --- This executes our policy over our predicted rougue from the NN
    local output = {}
    local N = #nnpreds
    -- Epsilon greedy strategy
    if torch.rand(1)[1] <= epsilon then  
        print('executing random policy')
        output = torch.totable(torch.round(torch.rand(N)))
    else     --- This is the action choice 1 select, 0 skip
        print('executing deterministic policy')
        for i =1, N do
            output[i] = (nnpreds[i][1] > nnpreds[i][2]) and 1 or 0
        end
    end
    return output
end

function score_model(pred, sentence_xs, epsilon, thresh, skip_rate)
    local pred = policy(pred, epsilon)
    local opt_action = {}
    local f1_t1, r1_t1, p1_t1 = 0., 0., 0.
    local f0_t1, r0_t1, p0_t1 = 0., 0., 0.
    local fscores, rscores, pscores = {}, {}, {}
    local fscores1, rscores1, pscores1 = {}, {}, {}
    local fscores0, rscores0, pscores0 = {}, {}, {}
    for i=1, #pred do
        --- This is the argmax part()
        opt_action[i] = (pred[i][1]  > pred[i][2]) and 1 or 0
        local curr_summary= buildPredSummary(geti_n(opt_action, 1, i), 
                                           geti_n(sentence_xs, 1, i),  nil) 
        fscores[i] = rougeF1({curr_summary[i]}, nuggets )
        rscores[i] = rougeRecall({curr_summary[i]}, nuggets )
        pscores[i] = rougePrecision({curr_summary[i]}, nuggets )

        if opt_action[i]==1 then
            fscores1[i] = threshold(fscores[i] - f1_t1, thresh)
            rscores1[i] = threshold(rscores[i] - r1_t1, thresh)
            pscores1[i] = threshold(pscores[i] - p1_t1, thresh)
 
            fscores0[i] = threshold(0. - f0_t1, thresh)
            rscores0[i] = threshold(0. - r0_t1, thresh)
            pscores0[i] = threshold(0. - p0_t1, thresh)
            -- if skip_rate = 0 we'll always run this
            -- if skip_rate = 1 we'll always skip it
            if skip_rate <= torch.rand(1)[1] then  
                f1_t1, r1_t1, p1_t1  = fscores1[i], rscores1[i], pscores1[i]
                f0_t1, r0_t1, p0_t1  = fscores0[i], rscores0[i], pscores0[i]
            end
        else 
            fscores1[i] = threshold(0. - f1_t1, thresh)
            rscores1[i] = threshold(0. - r1_t1, thresh)
            pscores1[i] = threshold(0. - p1_t1, thresh)

            fscores0[i] = threshold(fscores[i] - f0_t1, thresh)
            rscores0[i] = threshold(rscores[i] - r0_t1, thresh)
            pscores0[i] = threshold(pscores[i] - p0_t1, thresh)
            if skip_rate <= torch.rand(1)[1]  then  
                f1_t1, r1_t1, p1_t1  = fscores1[i], rscores1[i], pscores1[i]
                f0_t1, r0_t1, p0_t1  = fscores0[i], rscores0[i], pscores0[i]
            end
        end 
    end
    local labels = Tensor(rscores1):cat(Tensor(rscores0), 2)
    -- local labels = Tensor(fscores1):cat(Tensor(fscores0), 2)
    return labels, opt_action
end

function optimalPred(pred)
    local N = #pred
    local output = {}
    for i =1, N do
        output[i] = (pred[i][1] > pred[i][2]) and pred[i][1] or pred[i][2]
    end
    return output
end

function score_model2(pred, sentence_xs, epsilon)
    local actions = policy2(pred, epsilon)
    local pred = optimalPred(pred)
    local f1_t1, r1_t1, p1_t1 = 0., 0., 0.
    local f0_t1, r0_t1, p0_t1 = 0., 0., 0.
    local fscores, rscores, pscores = {}, {}, {}
    for i=1, #pred do
        --- This is the argmax part()
        local curr_summary= buildPredSummary(geti_n(actions, 1, i), 
                                           geti_n(sentence_xs, 1, i),  nil) 
        fscores[i] = rougeF1({curr_summary[i]}, nuggets ) - f1_t1
        rscores[i] = rougeRecall({curr_summary[i]}, nuggets ) - r1_t1
        pscores[i] = rougePrecision({curr_summary[i]}, nuggets ) - p1_t1
    end
    return pred, Tensor(fscores), actions
end

function build_bowmlp(vocab_size, embed_dim)
    local model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, embed_dim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse

   return model
end

function build_lstm(vocab_size, embed_dim)
    local model = nn.Sequential()
    :add(nn.LookupTableMaskZero(vocab_size, embed_dim)) -- will return a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, embed_dim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(embed_dim, embed_dim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(embed_dim, embed_dim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
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


    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)

    local FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add(nn.Linear(embed_dim * 3, outputSize) )
    FinalMLP:add(nn.Tanh())

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end
function build_model2(model, vocab_size, embed_dim, outputSize, use_cuda)
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
    mod4:add(nn.ReLU())

    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)
    ParallelModel:add(mod4)

    local FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add(nn.Linear(embed_dim * 4, outputSize) )
    FinalMLP:add(nn.Tanh())

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end

function getIndices(xtable, xindices)
    output = {}
    for i=1, #xindices do
        output[i] = xtable[xindices[i]]
    end
    return output
end

--- To do list:
    --- 1. Replicate node module 
        -- looked into this a little tried testing it wasn't straightforward
        -- NOT DONE
    --- 2. Change sumary to output Last K terms, not limited by sequence length
            -- Done, no longer adding zero for sequence
    --- 3. Output 2 score for rougue, 1 for action =1 and action = 0  
            -- Done but need to discuss htis more
    --- 4. share weights and embeddings between LSTMs
            -- NOT DONE 
    --- 5. threshold applied to rougue delta
            -- Done, should find some way to decide this intuitively
    --- 6. RMS prop in optim package
            -- Started looking into this, will require more review
    --- 7. adjust sampling methology and backpropogation 
            -- Done--ish, only sample through rows as I iterate through queries
                -- Technically should sample from each query and then the rows
    --- 8. Map tokens below some threshold to unknown -- Need to modify inputs
            --- NOT DONE
    --- 9. Not meant to do but did anyways: trying a sampling method to skip
function iterateModelQueries(input_path, query_file, batch_size, nepochs, inputs, 
                            model, crit, thresh, embed_dim, epsilon, delta, 
                            base_explore_rate, print_every,
                            learning_rate, J_sentences, K_tokens, use_cuda,
                            skiprate)
    --- This function iterates over the epochs, queries, and mini-batches to learn the model
    --- This version differs in that we output 2 units from the MLP and only the 3 LSTMs
    --- and simply map {0 - action_{t-1}} as the outcome that wasnt't selected
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

    print_string = string.format(
        "training model with learning rate = %.3f, K = %i, J = %i, and minibatch size = %i...",
                learning_rate, K_tokens, J_sentences, batch_size
                )

    print(print_string)

    vocab_size = 0
    maxseqlen = 0
    maxseqlenq = getMaxseq(query_file)

    action_query_list = {}
    yrougue_query_list = {}
    pred_query_list = {}

    for query_id = 1, #inputs do
        input_fn = inputs[query_id]['inputs']
        nugget_fn = inputs[query_id]['nuggets']

        input_file = csvigo.load({path = input_path .. input_fn, mode = "large", verbose = false})
        nugget_file = csvigo.load({path = input_path .. nugget_fn, mode = "large", verbose = false})

        vocab_sized = getVocabSize(input_file)
        vocab_sizeq = getVocabSize(query_file)
        vocab_size = math.max(vocab_size, vocab_sized, vocab_sizeq)

        maxseqlend = getMaxseq(input_file)
        maxseqlen = math.max(maxseqlen, maxseqlenq, maxseqlend)
        action_list = torch.totable(torch.round(torch.rand(#input_file)))
        action_list[1] = 0

        --- initialize the query specific lists
        action_query_list[query_id] = action_list
        yrougue_query_list[query_id] = torch.totable(torch.randn(#input_file, 2)) --- Actual
        pred_query_list[query_id] = torch.totable(torch.zeros(#input_file, 2))    --- Predicted
    end

    model  = build_model(model, vocab_size, embed_dim, 2, use_cuda)

    for epoch=0, nepochs, 1 do
        loss = 0.                    --- Compute a new MSE loss each time
        --- Looping over each bach of sentences for a given query
        for query_id = 1, #inputs do

            --- Grabbing all of the input data
            qs = inputs[query_id]['query']
            input_file = csvigo.load({path = input_path .. inputs[query_id]['inputs'], mode = "large", verbose = false})
            nugget_file = csvigo.load({path = input_path .. inputs[query_id]['nuggets'], mode = "large", verbose = false})
            nuggets = buildTermDocumentTable(nugget_file, nil)
            
            --- Building table of all of the input sentences
            local xtdm  = buildTermDocumentTable(input_file, K_tokens)
            
            --- Extracting the query specific summaries, actions, and rougue
            action_list = action_query_list[query_id]
            yrouge = yrougue_query_list[query_id] 
            preds = pred_query_list[query_id] 

            local nbatches = torch.floor( #input_file / batch_size)
            
            --- Initializing rouge metrics at time {t-1} and save scores, reset each new epoch
            local r_t1 , p_t1, f_t1 = 0., 0., 0.
            local rsm, psm, fsm = 0., 0., 0.
            local den = 0.
            for minibatch = 1, nbatches do
                if minibatch == 1 then
                    -- Need to skip the first row because it says "Text"
                    nstart = 2
                    nend = torch.round(batch_size * minibatch)
                end
                if minibatch > 1 and minibatch < nbatches then 
                    nstart = nend + 1
                    nend = torch.round(batch_size * minibatch)
                end
                if minibatch == nbatches then 
                    nstart = nend + 1
                    nend = #input_file
                end
                --- Processing the input data to get {query, input_sentences, summary, actions}
                local action_out = geti_n(action_list, nstart, nend)                
                local xout  = geti_n(xtdm, nstart, nend)    --- Extracting the mini-batch from our input sentences
                local xs  = padZeros(xout, K_tokens)    --- Padding the data by K tokens because we chose this as the max value
                local qs2 = padZeros({qs}, 5)
                local qrep = repeatTable(qs2[1], #xs)
                local sumry_list = buildPredSummary(action_out, xout, K_tokens * J_sentences)
                local sumry_ss = padZeros(sumry_list, K_tokens * J_sentences)
                --- Inserting data into tensors
                local summary = LongTensor(sumry_ss):t()
                local sentences = LongTensor(xs):t()
                local query = LongTensor(qrep):t()
                local actions = Tensor(action_out):resize(#xs, 1)

                --- Forward pass to estimate expected rougue)
                local pred_rougue = model:forward({sentences, summary, query, actions})
                --- Note setting the skip_rate = 0 means no random skipping of delta calculation
                labels, opt_action = score_model(torch.totable(pred_rougue), xout, epsilon, thresh, skiprate)

                -- Updating our bookkeeping tables
                yrouge = updateTable(yrouge, torch.totable(labels), nstart)
                preds =  updateTable(preds, torch.totable(pred_rougue), nstart)
                action_list = updateTable(action_list, opt_action, nstart)
            end
            --- Rerunning on the scoring on the full data and rescoring cumulatively
            --- Execute policy and evaluation based on our E[ROUGUE] after all of the minibatches
                --- Notice that pred_rougue gives us our optimal action by returning
                ---  E[ROUGUE | Select ] > E[ROUGUE | Skip]
            predsummary = buildPredSummary(action_list, xtdm, nil)
            predsummary = predsummary[#predsummary]

            rscore = rougeRecall({predsummary}, nuggets)
            pscore = rougePrecision({predsummary}, nuggets)
            fscore = rougeF1({predsummary}, nuggets)

            --- Updating variables
            action_query_list[query_id] = action_list
            yrougue_query_list[query_id] = yrouge
            pred_query_list[query_id] = preds

            if (epoch % print_every)==0 then
                perf_string = string.format(
                    "Epoch %i, epsilon = %.3f, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}, query = %s", 
                    epoch, epsilon, sumTable(action_list), #action_list, rscore, pscore, fscore, inputs[query_id]['query_name']
                    )
                print(perf_string)
            end

            --- creating the indices we want
            local indices = {}
            for i=1, 100 do
                indices[i] = math.random(2, #xtdm)
            end
            --- Have to skip over stupid header
            local xout = getIndices(xtdm, indices)
            local action_out = getIndices(action_list, indices)
            local labels = Tensor(getIndices(yrouge, indices)):resize(#xout, 1)
            local pred_rougue = Tensor(getIndices(preds, indices)):resize(#xout, 1)

            local xs  = padZeros(xout, K_tokens)    --- Padding the data by K tokens because we chose this as the max value
            local qs2 = padZeros({qs}, 5)
            local qrep = repeatTable(qs2[1], #xs)

            local sumry_list = buildPredSummary(action_out, xs, K_tokens * J_sentences)
            local summary = LongTensor(padZeros(sumry_list, K_tokens * J_sentences)):t()

            --- Inserting data into tensors            
            local sentences = LongTensor(xs):t()
            local query = LongTensor(qrep):t()

            loss = loss + crit:forward(pred_rougue, labels)
            grads = crit:backward(pred_rougue, labels)
            model:zeroGradParameters()
            model:backward({sentences, summary, query}, grads)
            model:updateParameters(learning_rate)

        end
        if epsilon <= base_explore_rate then                --- and leaving a random exploration rate
            epsilon = base_explore_rate
        else 
            epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
        end
    end
    return model, summary_query_list, action_query_list, yrougue_query_list
end 

function iterateModelQueries2(input_path, query_file, batch_size, nepochs, inputs, 
                            model, crit, thresh, embed_dim, epsilon, delta, 
                            base_explore_rate, print_every,
                            learning_rate, J_sentences, K_tokens, use_cuda)
    --- This function iterates over the epochs, queries, and mini-batches to learn the model
    --- This version differs in that we output 1 unit from the MLP and have actions as an input
    --- We also score under the two different possible actions and model based on this
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

    print_string = string.format(
        "training model with learning rate = %.3f, K = %i, J = %i, and minibatch size = %i...",
                learning_rate, K_tokens, J_sentences, batch_size
                )

    print(print_string)

    vocab_size = 0
    maxseqlen = 0
    maxseqlenq = getMaxseq(query_file)

    action_query_list = {}
    yrougue_query_list = {}
    pred_query_list = {}

    for query_id = 1, #inputs do
        input_fn = inputs[query_id]['inputs']
        nugget_fn = inputs[query_id]['nuggets']

        input_file = csvigo.load({path = input_path .. input_fn, mode = "large", verbose = false})
        nugget_file = csvigo.load({path = input_path .. nugget_fn, mode = "large", verbose = false})

        vocab_sized = getVocabSize(input_file)
        vocab_sizeq = getVocabSize(query_file)
        vocab_size = math.max(vocab_size, vocab_sized, vocab_sizeq)

        maxseqlend = getMaxseq(input_file)
        maxseqlen = math.max(maxseqlen, maxseqlenq, maxseqlend)
        action_list = torch.totable(torch.round(torch.rand(#input_file)))
        action_list[1] = 0

        --- initialize the query specific lists
        action_query_list[query_id] = action_list
        yrougue_query_list[query_id] = torch.totable(torch.randn(#input_file)) --- Actual
        pred_query_list[query_id] = torch.totable(torch.zeros(#input_file))    --- Predicted
    end

    model  = build_model2(model, vocab_size, embed_dim, 1, use_cuda)

    for epoch=0, nepochs, 1 do
        loss = 0.                    --- Compute a new MSE loss each time
        --- Looping over each bach of sentences for a given query
        for query_id = 1, #inputs do
            --- Grabbing all of the input data
            qs = inputs[query_id]['query']
            input_file = csvigo.load({path = input_path .. inputs[query_id]['inputs'], mode = "large", verbose = false})
            nugget_file = csvigo.load({path = input_path .. inputs[query_id]['nuggets'], mode = "large", verbose = false})
            nuggets = buildTermDocumentTable(nugget_file, nil)
            
            --- Building table of all of the input sentences
            local xtdm  = buildTermDocumentTable(input_file, K_tokens)
            
            --- Extracting the query specific summaries, actions, and rougue
            action_list = action_query_list[query_id]
            yrouge = yrougue_query_list[query_id] 
            preds = pred_query_list[query_id] 

            local nbatches = torch.floor( #input_file / batch_size)
            
            --- Initializing rouge metrics at time {t-1} and save scores, reset each new epoch
            local r_t1 , p_t1, f_t1 = 0., 0., 0.
            local rsm, psm, fsm = 0., 0., 0.
            local den = 0.
            for minibatch = 1, nbatches do
                if minibatch == 1 then
                    -- Need to skip the first row because it says "Text"
                    nstart = 2
                    nend = torch.round(batch_size * minibatch)
                end
                if minibatch > 1 and minibatch < nbatches then 
                    nstart = nend + 1
                    nend = torch.round(batch_size * minibatch)
                end
                if minibatch == nbatches then 
                    nstart = nend + 1
                    nend = #input_file
                end
                --- Processing the input data to get {query, input_sentences, summary, actions}
                local action_out = geti_n(action_list, nstart, nend)                
                local xout  = geti_n(xtdm, nstart, nend)    --- Extracting the mini-batch from our input sentences
                local xs  = padZeros(xout, K_tokens)    --- Padding the data by K tokens because we chose this as the max value
                local qs2 = padZeros({qs}, 5)
                local qrep = repeatTable(qs2[1], #xs)

                local sumry_list = buildPredSummary(action_out, xout, J_sentences * K_tokens)                
                local summary = LongTensor(padZeros(sumry_list, J_sentences * K_tokens)):t()

                local sentences = LongTensor(xs):t()
                local query = LongTensor(qrep):t()
                local actions = Tensor(action_out):resize(#xs, 1)

                --- Forward pass to estimate expected rougue
                local pred_rougue0 = model:forward({sentences, summary, query, actions})

                --- Swap the actions
                actions = torch.abs(actions - 1):resize(#xs, 1)
                action_out = torch.totable(actions)
                local sumry_list = buildPredSummary(action_out, xout, J_sentences * K_tokens)                
                local summary = LongTensor(padZeros(sumry_list, J_sentences * K_tokens)):t()

                --- Score under oposite action
                local pred_rougue1 = model:forward({sentences, summary, query, actions})

                pred_rougue = Tensor(pred_rougue0):cat(Tensor(pred_rougue1), 2)
                pred_rougue = torch.totable(pred_rougue)

                pred_rougue, labels, opt_action = score_model2(pred_rougue, xout, epsilon)

                if minibatch==nbatches then
                    pred_rougue = Tensor(pred_rougue)
                    print('sizes are ')
                    print(string.format('summary %i x %i', summary:size()[1], summary:size()[2]))
                    print(string.format('sentences %i x %i', sentences:size()[1], sentences:size()[2]))
                    print(string.format('query %i x %i', query:size()[1], query:size()[2]))
                    print(string.format('actions %i x %i', actions:size()[1], actions:size()[2]))
                    print(string.format('labels %i', labels:size()[1]))
                    print(string.format('prediction %i', pred_rougue:size()[1]))
                    pred_rougue = torch.totable(pred_rougue)
                end

                -- loss = loss + crit:forward(pred_rougue, labels)
                -- grads = crit:backward(pred_rougue, labels)
                -- model:zeroGradParameters()
                -- model:backward({sentences, summary, query, actions}, grads)
                -- model:updateParameters(learning_rate)
                -- pred_rougue = torch.totable(pred_rougue)

                -- Updating our bookkeeping tables
                yrouge = updateTable(yrouge, torch.totable(labels), nstart)
                preds =  updateTable(preds, pred_rougue, nstart)
                action_list = updateTable(action_list, opt_action, nstart)
            end
            --- Updating variables
            action_query_list[query_id] = action_list
            yrougue_query_list[query_id] = yrouge
            pred_query_list[query_id] = preds

            --- Rerunning on the scoring on the full data and rescoring cumulatively
            --- Execute policy and evaluation based on our E[ROUGUE] after all of the minibatches
                --- Notice that pred_rougue gives us our optimal action by returning
                ---  E[ROUGUE | Select ] > E[ROUGUE | Skip]
            predsummary = buildPredSummary(action_list, xtdm, nil)
            predsummary = predsummary[#predsummary]

            rscore = rougeRecall({predsummary}, nuggets)
            pscore = rougePrecision({predsummary}, nuggets)
            fscore = rougeF1({predsummary}, nuggets)

            if (epoch % print_every)==0 then
                perf_string = string.format(
                    "Epoch %i, epsilon = %.3f, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}, query = %s", 
                    epoch, epsilon, sumTable(action_list), #action_list, rscore, pscore, fscore, inputs[query_id]['query_name']
                    )
                print(perf_string)
            end

            --- Have to skip over stupid header
            local xout = geti_n(xtdm, 2, #xtdm)
            local action_out = geti_n(action_list, 2, #xtdm)
            local pred_rougue = Tensor(geti_n(preds, 2, #xtdm))
            local labels = Tensor(geti_n(yrouge, 2, #xtdm))

            local xs  = padZeros(xout, K_tokens)    --- Padding the data by K tokens because we chose this as the max value
            local qs2 = padZeros({qs}, 5)
            local qrep = repeatTable(qs2[1], #xs)

            local sumry_list = buildPredSummary(action_out, xout, J_sentences * K_tokens)
            local summary = LongTensor(padZeros(sumry_list, J_sentences * K_tokens)):t()

            --- Inserting data into tensors            
            local sentences = LongTensor(xs):t()
            local query = LongTensor(qrep):t()
            local actions = Tensor(action_out):resize(#xs, 1)

            print('sizes are ')
            print(string.format('summary %i x %i', summary:size()[1], summary:size()[2]))
            print(string.format('sentences %i x %i', sentences:size()[1], sentences:size()[2]))
            print(string.format('query %i x %i', query:size()[1], query:size()[2]))
            print(string.format('actions %i x %i', actions:size()[1], actions:size()[2]))
            print(string.format('labels %i', labels:size()[1]))
            print(string.format('prediction %i', pred_rougue:size()[1]))
            -- We backpropagate our observed outcomes and after the full set of queries

            loss = loss + crit:forward(pred_rougue, labels)
            grads = crit:backward(pred_rougue, labels)
            model:zeroGradParameters()
            model:backward({sentences, summary, query, actions}, grads)
            model:updateParameters(learning_rate)
        end
        if (epsilon - delta) <= base_explore_rate then                --- and leaving a random exploration rate
            epsilon = base_explore_rate
        else 
            epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
        end
    end
    return model, summary_query_list, action_query_list, yrougue_query_list
end 