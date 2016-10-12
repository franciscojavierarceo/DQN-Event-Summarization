function policy(nnpreds, epsilon)
    --- This executes our policy over our predicted rougue from the NN
    local pred = {}
    local N = nnpreds:size()[1]
    -- Epsilon greedy strategy
    if torch.rand(1)[1] <= epsilon then  
        for i=1, N do
            pred[i] = (torch.rand(1)[1] > 0.50 ) and 1 or 0
        end
    else 
    --- This is the action choice 1 select, 0 skip
        for i=1, N do
            pred[i] = (nnpreds[i][2] > nnpreds[i][1]) and 1 or 0
        end
    end
    return pred
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

    local mod4 = nn.Sequential()
    mod4:add(nn.Linear(1, embed_dim))
    mod4:add(nn.ReLU())

    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)
    -- ParallelModel:add(mod4)

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


--- To do list:
    --- 1. Replicate node module
    --- 2. Change sumary to output Last K terms, not limited by sequence length
    --- 3. Output 2 score for rougue, 1 for action =1 and action = 0
    --- 4. share weights and embeddings between LSTMs
    --- 5. threshold appliedto rougue delta
    --- 6. RMS prop in optim package
    --- 7. adjust sampling methology and backpropogation
    --- 8. Map tokens below some threshold to unknown
    --- 
function iterateModelQueries(input_path, query_file, batch_size, nepochs, inputs, 
                            model, crit, thresh, embed_dim, epsilon, delta, 
                            base_explore_rate, print_every,
                            learning_rate, K_sentences, K_tokens, use_cuda)
    --- This function iterates over the epochs, queries, and mini-batches to learn the model
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
        "training model with learning rate = %.3f, K = %i, and minibatch size = %i...",
                learning_rate, K_tokens, batch_size
                )

    print(print_string)

    vocab_size = 0
    maxseqlen = 0
    maxseqlenq = getMaxseq(query_file)

    action_query_list = {}
    yrougue_query_list = {}
    summary_query_list = {}

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
        --- initialize the query level lists
        summary_query_list[query_id] = torch.totable(torch.zeros(#input_file))
        action_query_list[query_id] = action_list
        yrougue_query_list[query_id] = torch.totable(torch.randn(#input_file, 2))
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

            --- Extracting the query specific summaries, actions, and rougue
            ss_list = summary_query_list[query_id] 
            action_list = action_query_list[query_id]
            yrouge = yrougue_query_list[query_id] 

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
                --- Building table of all of the input sentences
                local xtdm  = buildTermDocumentTable(input_file, K_tokens)
                --- Extracting the mini-batch from our input sentences
                local xout  = geti_n(xtdm, nstart, nend)    
                --- Padding the data by K tokens because we chose this as the max value
                local xs  = padZeros(xout, K_tokens)
                local qs2 = padZeros({qs}, 5)
                local qrep = repeatTable(qs2[1], #xs)

                local sumry_list = buildPredSummary(action_out, xout, K_sentences)
                local sumry_ss = padZeros(sumry_list, K_sentences * K_tokens)

                --- Inserting data into tensors
                local summary = LongTensor(sumry_ss):t()
                local sentences = LongTensor(xs):t()
                local query = LongTensor(qrep):t()
                local actions = Tensor(action_out):resize(#xs, 1)

                if use_cuda then
                     actions =  actions:cuda()
                end

                --- Forward pass to estimate expected rougue)
                pred_rougue = model:forward({sentences, summary, query})

                if use_cuda then
                    pred_rougue = pred_rougue:double()
                end
                --- Execute policy based on our E[ROUGUE]
                    --- Notice that pred_rougue gives us our optimal action by returning
                        ---  select {1} if E[ROUGUE]  > thresh  or 
                        ---  skip   {0} if E[ROUGUE] <= thresh
                -- opt_action = policy(pred_rougue, epsilon)
                opt_action = {}
                local rscores, pscores, fscores = {}, {}, {}
                for i=1, #xs do
                    -- Now we evaluate our action through the critic/Oracle
                    --- Calculating rouge scores; Call get_i_n() to cumulatively compute rouge
                    if i==1 then 
                        opt_action.insert(1)
                    local curr_summarySel = buildPredSummary(geti_n(opt_action, 1, i), 
                                                       geti_n(xout, 1 , i), K_sentences)

                    end
                    rscores[i] = threshold(rougeRecall(curr_summary, nuggets, K_sentences) - r_t1, thresh)
                    pscores[i] = threshold(rougePrecision(curr_summary, nuggets, K_sentences) - p_t1, thresh)
                    fscores[i] = threshold(rougeF1(curr_summary, nuggets, K_sentences) - f_t1, thresh)
                    rsm, psm, fsm = rsm+rscores[i] + r_t1, psm + pscores[i] + p_t1, fsm + fscores[i] + f_t1
                    r_t1, p_t1, f_t1 = rscores[i], pscores[i], fscores[i]
                end

                local labels = torch.Tensor(pscores)

                if use_cuda then
                     labels = labels:cuda()
                     pred_rougue = pred_rougue:cuda()
                end

                -- We backpropagate our observed outcomes
                loss = loss + crit:forward(pred_rougue, labels)
                grads = crit:backward(pred_rougue, labels)
                model:zeroGradParameters()
                model:backward({sentences, summary, query, actions}, grads)
                model:updateParameters(learning_rate)        

                -- Updating our bookkeeping tables
                yrouge = updateTable(yrouge, pscores, nstart)
                action_list = updateTable(action_list, opt_action, nstart)

                --- Calculating last one to see actual last rouge, without delta
                local den = den + #xs
                rscore, pscore, fscore = rsm/den, psm/den, fsm/den
                if (epoch % print_every)==0 then
                    perf_string = string.format(
                        "Epoch %i, epsilon = %.3f, minibatch %i/%i, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}, query = %s", 
                        epoch, epsilon, minibatch, nbatches, sumTable(opt_action), #opt_action, rscore, pscore, fscore, inputs[query_id]['query_name']
                        )
                    print(perf_string)
                end
            end

            if epsilon <= base_explore_rate then                --- and leaving a random exploration rate
                epsilon = base_explore_rate
            else 
                epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
            end
            summary_query_list[query_id] = ss_list 
            action_query_list[query_id] = action_list
            yrougue_query_list[query_id] = yrouge
        end
    end
    return model, summary_query_list, action_query_list, yrougue_query_list
end 