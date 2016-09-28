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

function iterateModel(nbatches, nepochs, x, model, crit, epsilon, delta, mxl,
                    base_explore_rate, print_every, nuggets, learning_rate, K)
    local r_t1 , p_t1, f_t1 = 0., 0., 0. 
    for epoch=0, nepochs, 1 do
        loss = 0                    --- Compute a new MSE loss each time
        local batch_size = torch.round( #x / nbatches)
        for minibatch = 1, nbatches do
            if minibatch == 1 then     -- Need +1 skip the first row
                nstart = 2
                nend = torch.round(batch_size * minibatch)
            end
            if minibatch == #x then 
                nstart = nend + 1
                nend = #x
            end
            if minibatch > 1 and minibatch < #x then 
                nstart = nend + 1
                nend = torch.round(batch_size * minibatch)
            end
            --- This step is processing the data
            local x_ss  = geti_n(x, nstart, nend)
            local out  = grabNsamples(x_ss, 1, #x_ss)     --- Extracting N samples
            local xs = padZeros(out, mxl)                 --- Padding the data by the maximum length
            local inputs = torch.LongTensor(xs)           --- This is the correct format to input it
            local xsT = inputs:t()

            if epoch > 0 then
                myPreds = model:forward(xsT)
                loss = loss + crit:forward(myPreds, ys)
                grads = crit:backward(myPreds, ys)
                model:backward(xsT, grads)
                --We update parameters at the end of each batch
                model:updateParameters(learning_rate)
                model:zeroGradParameters()
            end

            preds = policy(myPreds, epsilon, #xs)
            --- Concatenating predictions into a summary
            predsummary = buildPredSummary(preds, xs)
            --- Initializing rouge metrics at time {t-1} and save scores
            local rscores, pscores, fscores = {}, {}, {}
            for i=1, N do
                --- Calculating rouge scores; Call get_i_n() to cumulatively compute rouge
                rscores[i] = rougeRecall(geti_n(predsummary, 1, i), nuggets, K) - r_t1
                pscores[i] = rougePrecision(geti_n(predsummary, 1, i), nuggets, K) - p_t1
                fscores[i] = rougeF1(geti_n(predsummary, 1, i), nuggets, K) - f_t1
                r_t1, p_t1, f_t1 = rscores[i], pscores[i], fscores[i]
            end
            --- Calculating last one to see actual last rouge
            rscore = rougeRecall(predsummary, nuggets, K)
            pscore = rougePrecision(predsummary, nuggets, K)
            fscore = rougeF1(predsummary, nuggets, K)

            if (epoch % print_every)==0 then
                 -- This line is useful to view the min and max of the predctions
                -- if epoch> 0 then 
                --     print(myPreds:min(), myPreds:max())
                -- end
                perf_string = string.format(
                    "Epoch %i, sumy = %i, loss = %.6f, {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                    epoch, sumTable(preds), loss, rscore, pscore, fscore
                    )
                print(perf_string)
            end
            --- Updating the labels
            ys = torch.Tensor(fscores)         
            epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
            if epsilon <= 0 then                --- leave a random exploration rate
                epsilon = base_explore_rate
            end
        end
    end
    return model
end 