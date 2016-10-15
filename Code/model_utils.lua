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
    --- 2. Change sumary to output Last K terms, not limited by sequence length == Done
    --- 3. Output 2 score for rougue, 1 for action =1 and action = 0  === kind of done
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
                     pred_rougue =  torch.totable(pred_rougue)
                end
                --- Execute policy based on our E[ROUGUE]
                    --- Notice that pred_rougue gives us our optimal action by returning
                        ---  E[ROUGUE | Select ] > E[ROUGUE | Skip]

                pred_rougue = policy(pred_rougue, epsilon)

                opt_action = {}
                f1_rougue_select,  re_rougue_select, pr_rougue_select = {}, {}, {}
                f1_rougue_skip, re_rougue_skip , pr_rougue_skip = {}, {}, {}
                fsel_t1, fskp_t1, rsel_t1, rskp_t1, psel_t1, pskp_t1 = 0., 0., 0., 0., 0., 0.

                f1_t1, f0_t1, r1_t1, r0_t1, p1_t1, p0_t1 = 0., 0., 0., 0., 0., 0.
                fscores, rscores, pscores = {}, {}, {}
                fscores1, rscores1, pscores1 = {}, {}, {}
                fscores0, rscores0, pscores0 = {}, {}, {}
                --- TODO map evaluation to zero for the actions we don't take
                for i=1, #pred_rougue do
                    --- This is the argmax part()
                    opt_action[i] = (pred_rougue[i][1]  > pred_rougue[i][2]) and 1 or 0
                    curr_summary= buildPredSummary(geti_n(opt_action, 1, i), 
                                                       geti_n(xout, 1, i),  nil) 
                    fscores[i] = rougeF1({curr_summary[i]}, nuggets )
                    rscores[i] = rougeRecall({curr_summary[i]}, nuggets )
                    pscores[i] = rougePrecision({curr_summary[i]}, nuggets )

                    if opt_action[i]==1 then
                        fscores1[i] = fscores[i] - f1_t1
                        rscores1[i] = rscores[i] - r1_t1
                        pscores1[i] = pscores[i] - p1_t1

                        fscores0[i] = 0 - f0_t1
                        rscores0[i] = 0 - r0_t1
                        pscores0[i] = 0 - p0_t1                
                        f1_t1, r1_t1, p1_t1  = fscores1[i], rscores1[i], pscores1[i]
                        f0_t1, r0_t1, p0_t1  = fscores0[i], rscores0[i], pscores0[i]
                    else 
                        fscores1[i] = 0 - f1_t1
                        rscores1[i] = 0 - r1_t1
                        pscores1[i] = 0 - p1_t1

                        fscores0[i] = fscores[i] - f0_t1
                        rscores0[i] = rscores[i] - r0_t1
                        pscores0[i] = pscores[i] - p0_t1                
                        f1_t1, r1_t1, p1_t1  = fscores1[i], rscores1[i], pscores1[i]
                        f0_t1, r0_t1, p0_t1  = fscores0[i], rscores0[i], pscores0[i]
                    end 
                end
                    -- if i ==1 then 
                    --     curr_summarySkp = buildPredSummary({0}, {xout[i]},  nil)
                    --     curr_summarySel = buildPredSummary({1}, {xout[i]},  nil)
                    -- else
                    --     curr_summarySkp = buildPredSummary(tableConcat(geti_n(opt_action, 1, i-1), {0}), 
                    --                                         geti_n(xout, 1, i),  nil) 
                    --     curr_summarySel = buildPredSummary(tableConcat(geti_n(opt_action, 1, i-1), {1}), 
                    --                                         geti_n(xout, 1, i),  nil)
                    -- end
                    -- f1_rougue_select[i] = rougeF1({curr_summarySel[i]}, nuggets ) - fsel_t1
                    -- re_rougue_select[i] = rougeRecall({curr_summarySel[i]}, nuggets ) - rsel_t1
                    -- pr_rougue_select[i] = rougePrecision({curr_summarySel[i]}, nuggets ) - psel_t1
                    -- f1_rougue_skip[i] = rougeF1({curr_summarySkp[i]}, nuggets )  - fskp_t1
                    -- re_rougue_skip[i] = rougeRecall({curr_summarySkp[i]}, nuggets )  - rskp_t1
                    -- pr_rougue_skip[i] = rougePrecision({curr_summarySkp[i]}, nuggets )  - pskp_t1
                    -- fsel_t1, rsel_t1, psel_t1 = f1_rougue_select[i], pr_rougue_select[i], pr_rougue_select[i]
                    -- fskp_t1,  rskp_t1,  pskp_t1 = f1_rougue_skip[i], re_rougue_skip[i], pr_rougue_skip[i]
                    -- opt_action[i] = (f1_rougue_select[i] > f1_rougue_skip[i]) and 1 or 0
                -- end

                local labels = Tensor(fscores1):cat(Tensor(fscores0), 2)
                -- local labels = Tensor(f1_rougue_skip):cat(Tensor(f1_rougue_select), 2)

                if use_cuda then
                     labels = labels:cuda()
                     pred_rougue = Tensor(pred_rougue):cuda()
                end

                predsummary = buildPredSummary(opt_action, xout, nil)
                predsummary = predsummary[#predsummary]

                rscore = rougeRecall({predsummary}, nuggets)
                pscore = rougePrecision({predsummary}, nuggets)
                fscore = rougeF1({predsummary}, nuggets)

                -- We backpropagate our observed outcomes
                loss = loss + crit:forward(pred_rougue, labels)
                grads = crit:backward(pred_rougue, labels)
                model:zeroGradParameters()
                model:backward({sentences, summary, query}, grads)
                model:updateParameters(learning_rate)        

                -- Updating our bookkeeping tables
                action_list = updateTable(action_list, opt_action, nstart)
                yrouge = updateTable(yrouge, torch.totable(labels), nstart)

                --- Calculating last one to see actual last rouge, without delta
                -- local den = den + #xs
                -- rscore, pscore, fscore = rsm/den, psm/den, fsm/den
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