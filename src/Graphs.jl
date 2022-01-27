

#=========================================================
    LIB FOR NODES
=========================================================#


mutable struct Node

    x::Dict
    children::Union{Vector{Node}, Nothing}

end


mutable struct Functor
    
    fct::Function
    meta::Dict{Symbol, Vector{String}}
    
end


function get_node(id::Int64, df_Nodes::DataFrame)

    filter(row -> row.Id == id, df_Nodes)[1,:] |>
        (x -> Dict(pairs(x)))

end


function get_node(interfaceName::String, df_Nodes::DataFrame)

    filter(row -> row.Label == interfaceName, df_Nodes)[1,:] |>
        (x -> Dict(pairs(x)))

end


function get_children_ids(id::Int64, df_Arrows::DataFrame)

     df_Arrows[df_Arrows[!,"Line Source"] .== id, :]

end


function createTree(node::Union{Node, Nothing},
                    df_Nodes::DataFrame,
                    df_Arrows::DataFrame)

    if node != nothing

        id = node.x[:Id]
        df_ids = get_children_ids(id, df_Arrows)
        node.x[:map] = size(df_ids.Label)[1] > 1 ?
                Dict(map(x -> x[2] => x[1], enumerate(df_ids.Label))) :
                nothing
        

        if !haskey(dicCODE, node.x[:Label])
            
            @warn "unimplemented Node:: Id", node.x[:Id], " Name: " , node.x[:Label]
            
        end

        if size(df_ids)[1] > 0
        
            node.children = map(ix ->
                        createTree(Node(get_node(ix, df_Nodes), nothing),
                                   df_Nodes, df_Arrows),
                                df_ids[!,"Line Destination"])
        end

    end

    node

end

