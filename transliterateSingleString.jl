

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Code.jl")
include("src/Agent.jl")


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
data = Dict{String, Any}(
            "word" => parsedArgs["farsi-text"],
            "pos" => parsedArgs["pos-tagging"],
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


# run agent
runAgent(graph, dicBRAINS, data) |>
    println
