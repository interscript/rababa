

function processNode(node::Node,
                     dicBRAINS::Dict{String, Node},
                     df_Nodes::DataFrame,
                     data::Union{Nothing, Any})

    command = node.x[:Label]
    
    if haskey(dicCODE, command)

        states = collect(keys(data))
        unrecFields =
               filter(s -> !(s in states), dicCODE[command].meta[:in])

        if unrecFields != String[]

            @error string("command :: ", command,
                          " fields not recognised:: ", unrecFields)
            return nothing
        end

        dicCODE[command].fct(data, dicBRAINS, df_Nodes)

    else

        @error string("command not found/built :: ", command)
        nothing

   end

end


function runAgent(node::Node,
                  dicBRAINS::Dict{String, Node},
                  df_Nodes::DataFrame,
                  data::Union{Nothing, Any})

    name = node.x[:Label]

    node =

        if haskey(dicBRAINS, name)

            @info "brain name ::> ", name

            if data["brain"] != name

                # run elsewhere in graph
                runAgent(dicBRAINS[name].children[1], dicBRAINS, df_Nodes, data)
                data["brain"] = name

            end

            # continue locally
            node.children[1]

        else
            
            @info "node::> ", name
            data = processNode(node, dicBRAINS, df_Nodes, data)

            if node.children == nothing

                return haskey(data, "res") ? data["res"] : data["word"]

            end

            if length(node.children) > 1
                
                state = data["state"]
                @info "response::> ", state
                id = node.x[:map][state]
                node.children[id]

            else

                node.children[1]

            end

        end
    

    @info "data::> ", data
    runAgent(node, dicBRAINS, df_Nodes, data)

end
