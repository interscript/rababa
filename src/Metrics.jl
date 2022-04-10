
function evaluation(trans_orig, trans_model, orig)

    l_bugs = []
    tp, fp = 0, 0

    for (i, (o, t)) in enumerate(zip(trans_orig, trans_model))

        l_orig = filter(s -> s != "", split(strip(o), (' ','_',',','.','!',':',';','?')))
        l_trans = filter(s -> s != "", split(strip(t), (' ','_',',','.','!',':',';','?')))
            
        # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # println(o)
        # println(l_orig)
        # println(l_trans)
        # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        correct = true
        for d in zip(l_orig, l_trans)
            
            if d[1] == d[2]
                tp += 1
            else
                fp += 1
                correct = false
            end
            # println(d[1] ," :: ", d[2], " :: ", d[1] == d[2])

        end

        if !correct
            push!(l_bugs, Dict("id" => i, "trans" => o, "trans_model" => t))
        end

    end

    println("accuracy: ", tp / (tp + fp))
    [d["id"] for d in l_bugs]

end
