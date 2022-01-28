

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
                
        println("dbg")
        @error string("command not found/built :: ", command)
        nothing
                
   end

end


function runAgent(node::Node, 
                  dicBRAINS::Dict{String, Node}, 
                  data::Union{Nothing, Any})

    name = node.x[:Label]
    @info "node::> ", name
        
    node = 
    
        if haskey(dicBRAINS, node.x[:Label])
            
            @info "brain switch ::> ", node.x[:Label]
            node.children[1]
            
        else
            
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

