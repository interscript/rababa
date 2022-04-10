

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Rules.jl")
include("src/Agent.jl")
include("src/Metrics.jl")

using Logging
Logging.disable_logging(Logging.Info)

using PyCall

hazm = pyimport("hazm")

PATH_HAZM = "resources/postagger.model"
PATH_FLEXICON = "resources/"


stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM)


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the train model"

        "--file-name"
            help = "file-name to be transliterated \n
                    excepted if file-name = data/test.csv \n
                    in that case, tests are performed."

    end

    parse_args(s)

end


# parse commands
parsedArgs = parse_commandline()


# load brain data
data = deserialize(parsedArgs["path-model"])

entryBrain = data[:entry]
dicBRAINS = data[:dicBrains]
df_Nodes = data[:df_Nodes]
graph = dicBRAINS[entryBrain]


# prepare data
data = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "pre_pos" => nothing,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


m = 1
if parsedArgs["file-name"] in ["data/test.csv", "test"] # Run the test


    df_Test = DataFrame(CSV.File("data/test_data.csv")) #_data.csv"))

    df_Test[!,"transModel"] =
        map(d -> d |>
            py"""normalise""" |>
                hazm.word_tokenize |>
                    tagger.tag |>
                        (D -> map(d -> (dd = copy(data); 
                                        dd["word"] = d[1];
                                        dd["pos"] = d[2];
                                        dd["state"] = nothing;
                            try
                        
                                runAgent(graph, dicBRAINS, df_Nodes, dd);
                        
                            catch
                        
                                println("DBG:: ", dd["word"], " : ", dd["pos"]);
                                dd["word"]
                        
                            end
                ), D)) |>
                            (L -> join(L, " ")),
    df_Test[!,"orig"])

    ids = evaluation(df_Test[!, "trans"], df_Test[!, "transModel"], df_Test[!, "orig"])

    df_Bugs = df_Test[ids,:]

    println("error summary in: tests/test_debug.csv")
    CSV.write("tests/test_debug.csv", df_Bugs)


else # transliterate the file


    readlines(parsedArgs["file-name"], keep=true) |>
      (D ->
        map(d -> d |>
            py"""normalise""" |>
                hazm.word_tokenize |>
                    tagger.tag |>
                        (D -> map(d -> (data["word"] = d[1];
                                        data["pos"] = d[2];
                                        data["state"] = nothing;
                               runAgent(graph, dicBRAINS, df_Nodes, data)), D)) |>
                            (L -> join(L, " ")) |>
                                println,
            D))

end
