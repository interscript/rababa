
#==================================================================================
    HAZMing Interface to do operations via voice
==================================================================================#

# ran from humbaba.jl/run
DIR = JSON.parsefile("../config.json")["path"]
dataPATH = "$DIR/data/"
interfacesPATH = "$DIR/run/interfaces/"
projPATH = "$DIR/"
PATH_HAZM = "$DIR/run/interfaces/hazm/resources/postagger.model"

using PyCall

hazm = pyimport("hazm")
stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM)


dicHAZM = Dict()

dicHAZM["pos *"] = q -> Dict("HAZM" => string("pos ", q),
                           "data" =>
                           Dict("pos" => tagger.tag(hazm.word_tokenize(normalizer.normalize(q)))))

dicHAZM["stem *"] = q -> Dict("HAZM" => string("stem ", q),
                              "data" => Dict("stem" => stemmer.stem(q)))

dicHAZM["lemma *"] = q -> Dict("HAZM" => string("lemma ", q),
                             "data" => Dict("lemma" => lemmatizer.lemmatize(q)))

dicHAZM["normalize *"] = q -> Dict("HAZM" => string("normalize ", q),
                                   "data" =>
                                   Dict("normalise" => normalizer.normalize(q)))

dicHAZM["process *"] = q -> Dict("HAZM" => string("process ", q),
                                 "data" =>
                                 Dict(#"normalise" => normalizer.normalize(q),
                                      "lemma" => lemmatizer.lemmatize(q),
                                       "stem" => stemmer.stem(q)))


dicHAZM["help *"] = requestStr -> Dict("HAZM" => "help",
            "data" =>
            Dict("HAZM" =>
                 Dict("commando list" => collect(keys(dicHAZM))),
                      "library" => "Python library for digesting Persian text.",
                      "url" => "https://github.com/sobhe/hazm"))


struct InterfaceHAZM <: Interface

    data::Dict{Symbol, Any}

end


generateClassHAZM(s::String) = strip(s)


function buildDataHAZM(brains, side::String, dims)

    data = Dict{Symbol, Any}(:api => dicHAZM)
    data[:createClassFct] = s -> generateClassHAZM(s)
    InterfaceHAZM(data)

end


process(interface::InterfaceHAZM, str) = (k=sort(collect(keys(interface.data[:api])),
                                                 by=s -> compare(str, s, Levenshtein()),
                                                 rev=true)[1];
                                          Base.invokelatest(interface.data[:api][k], str))


process(interface::InterfaceHAZM, pingpong, dims) = (dim=dims[1];
                                                     str=pingpong[dim];
                                                     merge(pingpong,
                                                           try
                                                               process(interface, str)
                                                           catch
                                                               pingpong
                                                           end))


postprocess(interface::InterfaceHAZM, pingpong, dims) = (dim=dims[1];
                                                         pingpong[dim] = pingpong["HAZM"];
                                                         delete!(pingpong, "HAZM");
                                                         pingpong)


test(interfaceHAZM) = @test true
