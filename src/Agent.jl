

function processNode(node::Node, data::Union{Nothing, Any})

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

        dicCODE[command].fct(data)

    else

        @error string("command not found/built :: ", command)
        nothing

   end

end


function runAgent(node::Node,
                  dicBRAINS::Dict{String, Node},
                  data::Union{Nothing, Any})

    name = node.x[:Label]


    node =

        if haskey(dicBRAINS, name)

            @info "brain name ::> ", name

            if data["brain"] != name

                # run elsewhere in graph
                runAgent(dicBRAINS[name].children[1], dicBRAINS, data)
                data["brain"] = name

            end

            # continue locally
            node.children[1]

        else
            
            @info "node::> ", name
            data = processNode(node, data)

            if node.children == nothing

                return haskey(data,"res") ? data["res"] : data["word"]

            end

            if length(node.children) > 1

                id = node.x[:map][data["state"]]
                node.children[id]

            else

                node.children[1]

            end

        end

    runAgent(node, dicBRAINS, data)

end
