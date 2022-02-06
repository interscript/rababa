

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Code.jl")
include("src/Agent.jl")
include("src/hazm/get_assets.jl") # for PoS processing


using PyCall

hazm = pyimport("hazm")

PATH_HAZM = "resources/postagger.model"
PATH_FLEXICON = "resources/"

stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM);



function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--path-model"
            help = "path to the train model"
        "--farsi-text"
            help = "farsi text to be transliterated"
        "--pos-tagging"
                help = "PoS tagging, as found by hazm POSTagger"

    end

    parse_args(s)

end


# parse commands
parsedArgs = parse_commandline()


# load brain data
data = deserialize(parsedArgs["path-model"])

entryBrain = data[:entry]
dicBRAINS = data[:dicBrains]
graph = dicBRAINS[entryBrain]


# prepare data

pos = parsedArgs["pos-tagging"] |> 
    (p -> string(uppercase(p[1]), p[2:end]))

l_supported_POS = vcat(py"""l_PoS""", 
                       collect(keys(py"""d_map_FLEXI""")),
                       collect(keys(py"""d_map_HAZM""")))

if !(pos in l_supported_POS)
    
    @error "pos unrecognised, needs to be within: ", l_supported_POS
    exit()
        
else

    if pos in collect(keys(py"""d_map_FLEXI"""))
       
        pos = py"""d_map_FLEXI"""[pos]
        
    elseif pos in collect(keys(py"""d_map_HAZM"""))
        
        pos = py"""d_map_HAZM"""[pos]
        
    end
       
end


data = Dict{String, Any}(
            "word" => parsedArgs["farsi-text"],
            "pos" => pos,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


# run agent
runAgent(graph, dicBRAINS, data) |>
    println
