

include("hazm/py_code.jl")


dicCODE = Dict{String, Functor}()


# transliterator

dicCODE["change all instances of ي and ك in the text to ی and ک"] =
    Functor(d -> (d["word"]=py"""normalise"""(d["word"]); d),
            Dict(:in => ["word"], :out => ["word"]))

dicCODE["is the word found in the db?"] =
    Functor(d -> (d["data"]=py"""search_db"""(d["word"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["word", "pos"], :out => ["data", "state"]))

dicCODE["is it a verb?"] =
    Functor(d -> (d["state"] = d["pos"] == "V" ? "yes" : "no"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["lemmatize it!"] =
    Functor(d -> (d["lemma"] = lemmatizer.lemmatize(d["word"]); d),
            Dict(:in => ["word"], :out => ["lemma"]))

dicCODE["includes underscores?"] =
    Functor(d -> (d["state"] = contains(d["lemma"], "_") ? "yes" : "no"; d),
            Dict(:in => ["lemma"], :out => ["state"]))

dicCODE["does only one of the verb roots exist in the verb?"] =
    Functor(d -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 1 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))

dicCODE["output it!"] =
    Functor(d -> d,
            Dict(:in => ["word"], :out => []))

dicCODE["collision?"] =
    Functor(d -> (d["state"] = length(d["data"]) == 1 ? "no" : "yes"; d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["output its transliteration!"] =
    Functor(d -> (d),
            Dict(:in => ["data"], :out => ["res"]))

dicCODE["stem it!"] =
    Functor(d -> (d["stem"] = stemmer.stem(d["word"]); d),
            Dict(:in => ["word"], :out => ["stem"]))

dicCODE["is the verb root found in the db?"] =
    Functor(d -> (d["data"]=py"""search_db"""(d["lemma"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["lemma", "pos"], :out => ["data", "state"]))

dicCODE["does the root of the word exist in the database?"] =
    Functor(d -> (d["data"]=py"""search_db"""(d["stem"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["stem", "pos"], :out => ["data", "state"]))


# collision-handler

dicCODE["is there an instance of the word with the desired pos?"] =
    Functor(d -> ( #println(d["data"]);
            d["state"] = py"""has_entries_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["is there only one instance of the word with the desired pos?"] =
    Functor(d -> (d["state"] = py"""has_only_one_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["return the transliteration of the instance with the desired pos!"] =
    Functor(d -> (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); println(d["res"]);d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the desired pos that has the highest frequency!"] =
    Functor(d -> (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the highest frequency!"] =
    Functor(d -> (d["res"] = py"""return_highest_search"""(d["data"]); d),
            Dict(:in => ["data"], :out => ["res"]))
