

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Code.jl")
include("src/Agent.jl")
include("src/Metrics.jl")


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
            "word" => nothing,
            "pos" => nothing,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


df_Test = DataFrame(CSV.File("data/test.csv"))

df_Test[!,"transModel"] =
    map(d -> d |>
            py"""normalise""" |>
                hazm.word_tokenize |>
                    tagger.tag |>
                        (D -> map(d -> (data["word"] = d[1];
                                        data["pos"] = d[2];
                                        data["state"] = nothing;
                               runAgent(graph, dicBRAINS, data)), D)) |>
                            (L -> join(L, " ")),
        df_Test[!,"orig"]);


ids = evaluation(df_Test[!, "trans"], df_Test[!, "transModel"], df_Test[!, "orig"])

df_Bugs = df_Test[ids,:]

println("error summary in: tests/test_debug.csv")
CSV.write("tests/test_debug.csv", df_Bugs);
