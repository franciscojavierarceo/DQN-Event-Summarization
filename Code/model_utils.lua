function score_model(opt_action, sentence_xs, input_nuggets, thresh, skip_rate, metric)
    --- Scores the model given the list of optimal actions, input sentences, and nuggets
        --- Note: we calculate *change* in rougue from time t-1 to t for each sentence
        --- The skip_rate controls this delta calculation
            --- skip_rate == 0 turns skip_rate off ==> always updates the lag 
                --- means we are computing s_t - s_{t-1}
            --- skip_rate >= 1 turns skip_rate always ==> never updates the lag
                --- means we are computing s_t - s_{t_0} and s_{t_0} == 0 ==> s_t
    local s_t1 = 0.
    local scores = {}
    if metric=='f1' then
        eval_func = rougeF1
    end
    for i=1, #opt_action do
        local curr_summary= buildPredSummary(geti_n(opt_action, 1, i), 
                                           geti_n(sentence_xs, 1, i),  nil)

        scores[i] = threshold(eval_func({curr_summary[i]}, input_nuggets ) - s_t1, thresh)
        s_t1 = scores[i]
        --- Skip rate controls how often we skip updating the lag
        --- e.g., if skip_rate = 0.25 <= [0, 1] ==> update lag 75% of the time
        if skip_rate > 0 then 
            if skip_rate <= torch.rand(1)[1] then
                s_t1 = scores[i] + s_t1
            end 
        else 
            s_t1 = scores[i] + s_t1
        end
    end
    return scores
end

function build_bowmlp(nn_vocab_module, edim)
    local model = nn.Sequential()
    :add(nn_vocab_module)           -- returns a (sequence-length x batch-size x edim) tensor
    :add(nn.Sum(1, edim, true))     -- splits into a sequence-length table with (batch-size x edim) entries
    :add(nn.Linear(edim, edim))     -- map last state to a score for classification
    :add(nn.Tanh())                 --     :add(nn.ReLU()) <- this one did worse
   return model
end

function build_lstm(nn_vocab_module, edim)
    local model = nn.Sequential()
    :add(nn_vocab_module)            -- returns a (sequence-length x batch-size x edim) tensor
    :add(nn.SplitTable(1, edim))     -- splits into a sequence-length table with (batch-size x edim) entries
    :add(nn.Sequencer(nn.LSTM(edim, edim)))
    :add(nn.SelectTable(-1))            -- selects last state of the LSTM
    :add(nn.Linear(edim, edim))         -- map last state to a score for classification
    :add(nn.Tanh())                     --     :add(nn.ReLU()) <- this one did worse
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

    mod2 = mod1:clone()         --- Cloning the first model to share the weights
    mod3 = mod1:clone()         --- across the different inputs

    local ParallelModel = nn.ParallelTable()
    :add(mod1)                  --- Adding in the parts of the model
    :add(mod2)                  --- for each of the 3 inputs
    :add(mod3)                  --- i.e., sentence, summary, query

    local FinalMLP = nn.Sequential()
    :add(ParallelModel)         --- Adding in the components
    :add(nn.JoinTable(2))       --- Joining the components back together
    :add(nn.Linear(embed_dim * 3, 2) )      --- Adding linear layer to output 2 units
    :add(nn.Max(2) )            --- Max over the 2 units (action) dimension
    :add(nn.Tanh())             --- Adding a non-linearity

    if use_cuda then
        return FinalMLP:cuda()
    else
        return FinalMLP
    end
end

--- To do list:
    --- 1. RMS prop in optim package
            -- NOT DONE
    --- 2. Map tokens below some threshold to unknown -- Need to modify inputs
            -- NOT DONE

function iterateModelQueries(input_path, query_file, batch_size, nepochs, inputs, 
                            nn_model, crit, thresh, embed_dim, epsilon, delta, 
                            base_explore_rate, print_every,
                            learning_rate, J_sentences, K_tokens, use_cuda,
                            skiprate, emetric)
    --- This function iterates over the epochs, queries, and sentences to learn the model
    if use_cuda then
        Tensor = torch.CudaTensor
        LongTensor = torch.CudaLongTensor
        crit = crit:cuda()
        print("...running on GPU")
    else
        torch.setnumthreads(8)
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

    --- Initializing query book-keeping 
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

    --- Specify model
    model  = build_model(nn_model, vocab_size, embed_dim, use_cuda)

    --- This is the main training function
    for epoch=0, nepochs, 1 do
        loss = 0.                    --- Compute a new MSE loss each time
        --- Looping over each bach of sentences for a given query
        for query_id = 1, #inputs do
            --- Grabbing all of the input data
            qs = inputs[query_id]['query']
            input_file = csvigo.load({path = input_path .. inputs[query_id]['inputs'], mode = "large", verbose = false})
            nugget_file = csvigo.load({path = input_path .. inputs[query_id]['nuggets'], mode = "large", verbose = false})
            
            --- Dropping the headers (which is the 1st row)
            input_file = geti_n(input_file, 2, #input_file) 
            nugget_file = geti_n(nugget_file, 2, #nugget_file) 
            
            --- Building table of indices for 
                -- first K_tokens of the input sentences
                -- and all tokens of the nuggets
            nuggets = buildTermDocumentTable(nugget_file, nil)
            xtdm  = buildTermDocumentTable(input_file, K_tokens)

            --- Extracting the query specific actions, labels, and predictions
            action_list = action_query_list[query_id]
            yrougue = yrougue_query_list[query_id] 
            preds = pred_query_list[query_id]
            
            --- Forward pass
            for minibatch = 1, #xtdm do
                --- Notice that the actionlist is optimized at after each iteration
                --- using geti_n to reduce computing time...though its still O(n) either way
                local summaries = padZeros(buildCurrentSummary(geti_n(action_list, 1, minibatch), 
                                                               geti_n(xtdm, 1, minibatch), 
                                        K_tokens * J_sentences), 
                                        K_tokens * J_sentences)
                --- Creating data into tensors
                sentence = LongTensor(padZeros( {xtdm[minibatch]}, K_tokens) ):t()
                summary = LongTensor({ summaries[minibatch] }):t()
                query = LongTensor( padZeros({qs}, 5) ):t()

                --- Retrieve intermediate optimal action in model.get(3).output
                local pred_rougue = model:forward({sentence, summary, query})   
                local pred_actions = torch.totable(model:get(3).output)         --- has 2 output units
                
                --- Epsilon greedy strategy
                if torch.rand(1)[1] < epsilon then 
                    opt_action = torch.round(torch.rand(1))[1]
                else 
                    --- Notice that pred_actions gives us our optimal action by returning
                    ---  E[ROUGUE | Select ] > E[ROUGUE | Skip]
                    opt_action = (pred_actions[1][1] > pred_actions[1][2]) and 1 or 0
                end 
                
                -- Updating book-keeping tables at sentence level
                preds[minibatch] = pred_rougue[1]
                action_list[minibatch] = opt_action
            end --- ends the sentence level loop
            --- Note setting the skip_rate = 0 means no random skipping of delta calculation
            yrougue = score_model(action_list, 
                            xtdm,
                            nuggets,
                            thresh, 
                            skiprate, 
                            emetric)

            --- Updating book-keeping tables at query level
            pred_query_list[query_id] = preds
            yrougue_query_list[query_id] = yrougue
            action_query_list[query_id] = action_list

            --- Rerunning the scoring on the full data and rescoring cumulatively
            --- Execute policy and evaluation based on our E[ROUGUE] after all of the minibatches
            predsummary = buildPredSummary(action_list, xtdm, nil)
            predsummary = predsummary[#predsummary]

            rscore = rougeRecall({predsummary}, nuggets)
            pscore = rougePrecision({predsummary}, nuggets)
            fscore = rougeF1({predsummary}, nuggets)

            --- creating randomly sampled query and input indices
            local qindices = {}
            local xindices = {}
            for i=1, batch_size do
                qindices[i] = math.random(1, #inputs)
                xindices[i] = math.random(1, #xtdm)
            end
            --- Building summaries on full set of input data then sampling after
            --- Need to do summaries first b/c if you build after sampling 
            --- you'll get incorrect summaries, also need to padZeros for empty summaries
            local summaries = padZeros(buildCurrentSummary(action_list, xtdm, 
                                        K_tokens * J_sentences), 
                                        K_tokens * J_sentences)
            --- Backward pass
            for i= 1, batch_size do
                sentence = LongTensor(padZeros( {xtdm[xindices[i]]}, K_tokens) ):t()
                summary = LongTensor({summaries[xindices[i]]}):t()
                query = LongTensor(padZeros({qs}, 5)):t()
                labels = Tensor({yrougue[xindices[i]]})
                pred_rougue = Tensor({preds[xindices[i]]})

                loss = loss + crit:forward(pred_rougue, labels)
                --- Backprop model 
                local grads = crit:backward(pred_rougue, labels)
                model:zeroGradParameters()
                --- For some reason runnign the :forward() makes the backward pass work
                --- spent a lot of time trying to debug why :backward() didn't work without it
                --- but I couldn't figure it out, then I tried this and it works...seems wrong.
                --- I'll ask Chris about this and see what he thinks
                local tmp = model:forward({sentence, summary, query})
                model:backward({sentence, summary, query}, grads)
                model:updateParameters(learning_rate)
            end
            if (epoch % print_every)==0 then
                perf_string = string.format(
                    "Epoch %i, loss  = %.3f, epsilon = %.3f, sum(y)/len(y) = %i/%i, {Recall = %.6f, Precision = %.6f, F1 = %.6f}, query = %s", 
                    epoch, loss, epsilon, sumTable(action_list), #action_list, rscore, pscore, fscore, inputs[query_id]['query_name']
                    )
                print(perf_string)
            end
        end -- ends the query level loop
        --- Reducing epsilon-greedy search linearly and setting it to the base rate
        if (epsilon - delta) <= base_explore_rate then
            epsilon = base_explore_rate
        else 
            epsilon = epsilon - delta
        end
    end -- ends the epoch level loop
    return model, summary_query_list, action_query_list, yrougue_query_list
end 