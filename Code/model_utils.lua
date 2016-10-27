function score_model(opt_action, sentence_xs, epsilon, thresh, skip_rate, metric)
    local f_t1, r_t1, p_t1 = 0., 0., 0.
    local fscores, rscores, pscores = {}, {}, {}
    for i=1, #opt_action do
        local curr_summary= buildPredSummary(geti_n(opt_action, 1, i), 
                                           geti_n(sentence_xs, 1, i),  nil) 
        fscores[i] = rougeF1({curr_summary[i]}, nuggets )
        rscores[i] = rougeRecall({curr_summary[i]}, nuggets )
        pscores[i] = rougePrecision({curr_summary[i]}, nuggets )        
        if skip_rate <= torch.rand(1)[1] then  
            f_t1, r_t1, p_t1 = fscores[i], rscores[i], pscores[i]
        end
    end
    return fscores
end


function build_bowmlp(nn_vocab_module, edim)
    local model = nn.Sequential()
    :add(nn_vocab_module)            -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.Sum(1, edim, true)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Linear(edim, edim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_lstm(nn_vocab_module, edim)
    local model = nn.Sequential()
    :add(nn_vocab_module)            -- returns a sequence-length x batch-size x embedDim tensor
    :add(nn.SplitTable(1, edim)) -- splits into a sequence-length table with batch-size x embedDim entries
    :add(nn.Sequencer(nn.LSTM(edim, edim)))
    :add(nn.SelectTable(-1)) -- selects last state of the LSTM
    :add(nn.Linear(edim, edim)) -- map last state to a score for classification
    :add(nn.Tanh())                     ---     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_model(model, vocab_size, embed_dim, use_cuda)
    local nn_vocab = nn.LookupTableMaskZero(vocab_size, embed_dim)
    if model == 'bow' then
        print("Running BOW model")
        mod1 = build_bowmlp(nn_vocab, embed_dim)
    end
    if model == 'lstm' then         
        print("Running LSTM model")
        mod1 = build_lstm(nn_vocab, embed_dim)
    end

    mod2 = mod1:clone()
    mod3 = mod1:clone()

    local ParallelModel = nn.ParallelTable()
    ParallelModel:add(mod1)
    ParallelModel:add(mod2)
    ParallelModel:add(mod3)

    local FinalMLP = nn.Sequential()
    FinalMLP:add(ParallelModel)
    FinalMLP:add(nn.JoinTable(2))
    FinalMLP:add(nn.Linear(embed_dim * 3, 2) )
    -- Taking the max over the second dimension
    FinalMLP:add(nn.Max(2) )
    FinalMLP:add(nn.Tanh())

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end

--- To do list:
    --- 1. Replicate node module 
        -- looked into this a little tried testing it wasn't straightforward
        -- NOT DONE
    --- 2. Change sumary to output Last K terms, not limited by sequence length
            -- Done, no longer adding zero for sequence
    --- 3. Output 2 score for rougue, 1 for action =1 and action = 0  
            -- Done but need to discuss this more
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
                            nn_model, crit, thresh, embed_dim, epsilon, delta, 
                            base_explore_rate, print_every,
                            learning_rate, J_sentences, K_tokens, use_cuda,
                            skiprate, emetric)
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
        "training model with metric = %s, learning rate = %.3f, K = %i, J = %i, threshold = %.3f, embedding size = %i",
                emetric, learning_rate, K_tokens, J_sentences, thresh, embed_dim, batch_size
                )

    print(print_string)

    vocab_size = 0
    maxseqlen = 0
    maxseqlenq = getMaxseq(query_file)

    action_query_list = {}
    yrougue_query_list = {}
    pred_query_list = {}

    --- Initializing query information
    for query_id = 1, #inputs do
        input_fn = inputs[query_id]['inputs']
        nugget_fn = inputs[query_id]['nuggets']

        input_file = csvigo.load({path = input_path .. input_fn, mode = "large", verbose = false})
        nugget_file = csvigo.load({path = input_path .. nugget_fn, mode = "large", verbose = false})
        input_file = geti_n(input_file, 2, #input_file) 
        nugget_file = geti_n(nugget_file, 2, #nugget_file) 

        vocab_sized = getVocabSize(input_file)
        vocab_sizeq = getVocabSize(query_file)
        vocab_size = math.max(vocab_size, vocab_sized, vocab_sizeq)

        maxseqlend = getMaxseq(input_file)
        maxseqlen = math.max(maxseqlen, maxseqlenq, maxseqlend)
        action_list = torch.totable(torch.round(torch.rand(#input_file)))

        --- initialize the query specific lists
        action_query_list[query_id] = action_list
        yrougue_query_list[query_id] = torch.totable(torch.randn(#input_file, 1)) --- Actual
        pred_query_list[query_id] = torch.totable(torch.zeros(#input_file, 1))    --- Predicted
    end

    model  = build_model(nn_model, vocab_size, embed_dim, use_cuda)

    for epoch=0, nepochs, 1 do
        loss = 0.                    --- Compute a new MSE loss each time
        --- Looping over each bach of sentences for a given query
        for query_id = 1, #inputs do
            --- Grabbing all of the input data
            qs = inputs[query_id]['query']
            input_file = csvigo.load({path = input_path .. inputs[query_id]['inputs'], mode = "large", verbose = false})
            nugget_file = csvigo.load({path = input_path .. inputs[query_id]['nuggets'], mode = "large", verbose = false})
            --- Dropping the headers
            input_file = geti_n(input_file, 2, #input_file) 
            nugget_file = geti_n(nugget_file, 2, #nugget_file) 
            
            --- Building table of all of the input sentences
            nuggets = buildTermDocumentTable(nugget_file, nil)
            xtdm  = buildTermDocumentTable(input_file, K_tokens)

            --- Extracting the query specific summaries, actions, and rougue
            action_list = action_query_list[query_id]
            yrougue = yrougue_query_list[query_id] 
            preds = pred_query_list[query_id]
            
            --- Loop over file to execute forward pass to estimate expected rougue
            for minibatch = 1, #xtdm do
                --- Notice that the actionlist is optimized at after each iteration
                local summaries = padZeros(buildCurrentSummary(action_list, xtdm, 
                                        K_tokens * J_sentences), 
                                        K_tokens * J_sentences)
                sentence = LongTensor(padZeros( {xtdm[minibatch]}, K_tokens) ):t()
                summary = LongTensor({ summaries[minibatch] }):t()
                query = LongTensor( padZeros({qs}, 5) ):t()

                --- Retrieve intermediate optimal action in model.get(3).output
                local pred_rougue = model:forward({sentence, summary, query})   
                local pred_actions = torch.totable(model:get(3).output)
                opt_action = (pred_actions[1][1] > pred_actions[1][2]) and 1 or 0
                
                -- Updating our book-keeping tables
                preds[minibatch] = pred_rougue[1]
                action_list[minibatch] = opt_action
            end
            --- Note setting the skip_rate = 0 means no random skipping of delta calculation
            yrougue = score_model(action_list, 
                            xtdm,
                            epsilon, 
                            thresh, 
                            skiprate, 
                            emetric)

            --- Updating variables
            pred_query_list[query_id] = preds
            yrougue_query_list[query_id] = yrougue
            action_query_list[query_id] = action_list

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

            --- creating the indices we want
            -- local qindices = {}
            local xindices = {}
            for i=1, batch_size do
                -- qindices[i] = math.random(1, #inputs)
                xindices[i] = math.random(1, #xtdm)
            end

            local summaries = padZeros(buildCurrentSummary(action_list, xtdm, 
                                        K_tokens * J_sentences), 
                                        K_tokens * J_sentences)
            --- Backward step
            for i= 1, batch_size do
                sentence = LongTensor(padZeros( {xtdm[xindices[i]]}, K_tokens) ):t()
                summary = LongTensor({summaries[xindices[i]]}):t()
                query = LongTensor(padZeros({qs}, 5)):t()

                labels = Tensor({yrougue[xindices[i]]})
                pred_rougue = Tensor({preds[xindices[i]]})

                --- Backprop model
                loss = loss + crit:forward(pred_rougue, labels)
                local grads = crit:backward(pred_rougue, labels)
                model:zeroGradParameters()
                --- For some reason doing this fixes it
                local tmp = model:forward({sentence, summary, query})
                model:backward({sentence, summary, query}, grads)
                model:updateParameters(learning_rate)
            end
        end -- ends the query loop
        if (epsilon - delta) <= base_explore_rate then                --- and leaving a random exploration rate
            epsilon = base_explore_rate
        else 
            epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
        end
    end
    return model, summary_query_list, action_query_list, yrougue_query_list
end 