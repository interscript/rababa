

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Agent.jl")
include("src/Code.jl")



function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--path-lucidchart-csv"
            help = "path to the graph csv"
        "--brain-entry"
            help = "brain entry"
        "--path-model"
            help = "path to the train model, e.g. resources/model.dat"

    end

    parse_args(s)

end


parsedArgs = parse_commandline()


graphName = parsedArgs["path-lucidchart-csv"]
modelName = parsedArgs["path-model"]
brainEntry = lowercase(parsedArgs["brain-entry"])
entryFound = false

# Parse csv data
df = DataFrame(CSV.File(graphName))
df[!,"Label"] = map(x -> ismissing(x) ? Missing : lowercase(x), df[!,"Text Area 1"])

df_Nodes = filter(row -> row.Name in ["Decision", "Process", "Terminator"], df)
df_Arrows = filter(row -> row.Name in ["Line"], df);
df_Brains = filter(row -> row.Name in ["Curly Brace Note"], df);


dicBRAINS = Dict{String, Node}()

brainsList = df_Brains[!, "Label"]


if !(brainEntry in brainsList)

    @warn "brain-entry not found in graph!
              (notice that lowercases of node names are taken)"
    exit()

end

for b in brainsList

    @info "build brain: ", b

    try

        dicBRAINS[b] = get_node(b, df_Brains) |>
                (D -> (n=Node(D, nothing); n.x[:depth]=0; n)) |>
                    (N -> createTree(N, df_Nodes, df_Arrows, df_Brains))

    catch

        @error "error! brain : ", b

    end

end


println("save data to: ", modelName)
serialize(modelName, Dict(:dicBrains => dicBRAINS,
                          :df_Nodes => df_Nodes,
                          :entry => brainEntry))
println("data saved to: ", modelName)
