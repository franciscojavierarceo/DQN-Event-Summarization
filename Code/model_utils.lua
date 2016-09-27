function iterateModel(nepochs, model, crit, inputs, 
                    xs, ys, epsilon, delta, 
                    base_explore_rate, print_every,
                    nuggets, learning_rate)
    local xsT = inputs:t()
    local r_t1 , p_t1, f_t1 = 0., 0., 0.      
    for epoch=0, nepochs, 1 do
        loss = 0                    --- Compute a new MSE loss each time
        if epoch > 0 then
            myPreds = model:forward(xsT)
            loss = loss + crit:forward(myPreds, ys)
            grads = crit:backward(myPreds, ys)
            model:backward(xsT, grads)        
            --We update params at the end of each batch
            model:updateParameters(learning_rate)
            model:zeroGradParameters()
        end
        
        preds = {}
        if torch.rand(1)[1] <= epsilon then  -- Epsilon greedy strategy
            for i=1, N do
                preds[i] = (torch.rand(1)[1] > 0.5 ) and 1 or 0
            end
        else 
            --- This is the action choice 1 select, 0 skip
            for i=1, N do
                preds[i] = (myPreds[i][1] > 0) and 1 or 0
            end
        end
        --- Concatenating predictions into a summary
        predsummary = buildPredSummary(preds, xs)
        --- Initializing rouge metrics at time {t-1} and save scores
        local rscores, pscores, fscores = {}, {}, {}
        for i=1, N do
            --- Calculating rouge scores; Call get_i_n() to cumulatively compute rouge
            rscores[i] = rougeRecall(geti_n(predsummary, i), nggs) - r_t1
            pscores[i] = rougePrecision(geti_n(predsummary, i), nggs) - p_t1
            fscores[i] = rougeF1(geti_n(predsummary, i), nggs) - f_t1
            r_t1, p_t1, f_t1 = rscores[i], pscores[i], fscores[i]
        end
        --- Calculating last one to see actual last rouge
        rscore = rougeRecall(predsummary, nggs)
        pscore = rougePrecision(predsummary, nggs)
        fscore = rougeF1(predsummary, nggs)

        if (epoch % print_every)==0 then
            if epoch> 0 then 
                print(myPreds:min(), myPreds:max(), c)
            end
            perf_string = string.format(
                "Epoch %i, sumy = %i, loss = %.6f, {Recall = %.6f, Precision = %.6f, F1 = %.6f}", 
                epoch, sumTable(preds), loss, rscore, pscore, fscore
                )
            print(perf_string)
        end
        ys = torch.Tensor(fscores)          --- Updating the labels
        epsilon = epsilon - delta           --- Decreasing the epsilon greedy strategy
        if epsilon <= 0 then                --- leave a random exploration rate
            epsilon = base_explore_rate
        end
    end
    return model
end 