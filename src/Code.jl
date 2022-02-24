

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
    Functor(d -> (d["state"] = d["pos"] == "Verb" ? "yes" : "no"; d),
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
    Functor(d -> (haskey(d, "res") ? d :
                   typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                        (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); d) :
                         d),
            Dict(:in => ["data"], :out => []))

dicCODE["collision?"] =
    Functor(d -> (d["state"] = length(d["data"]) == 1 ? "no" : "yes"; d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["output its transliteration!"] =
    Functor(d -> (haskey(d, "res") ? d :
                    typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                        (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); d) :
                         d),
            Dict(:in => [], :out => []))

dicCODE["return its transliteration!"] = dicCODE["output its transliteration!"]


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

dicCODE["transliterate each side of underscore separately in proper order"] =
    Functor(d -> split(d["lemma"], "_") |>
                (D -> map(x -> py"""return_highest_search_pos"""(x, d["pos"]), D) |> join),
            Dict(:in => ["lemma"], :out => ["res"]))


# collision-handler

dicCODE["is there an instance of the word with the desired pos?"] =
    Functor(d -> ( # println(d["data"]);
            d["state"] = py"""has_entries_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["is there only one instance of the word with the desired pos?"] =
    Functor(d -> (d["state"] = py"""has_only_one_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["return the transliteration of the instance with the desired pos!"] =
    Functor(d -> (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]);d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the desired pos that has the highest frequency!"] =
    Functor(d -> (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the highest frequency!"] =
    Functor(d -> (d["res"] = py"""return_highest_search"""(d["data"]); d),
            Dict(:in => ["data"], :out => ["res"]))


# Full Model

#===
dicCODE["find the longest substring of the input that exists in affixes and starts in the beginning of the input and run affix-handler on it then omit that substring from the input and do this again until the input is empty. then return the concatenation of all the returned transliterations."] = 
    Functor(, 
            Dict())

dicCODE["is there only one instance of the affix?"] =
    Functor(d -> ,
            Dict(:in => ["affix"], :out => ["state"]))
===#

dicCODE["return \"id\""] = 
    Functor(d -> d,
            Dict(:in => [], :out => []))

dicCODE["terminator"] = 
    Functor(d -> d,
            Dict(:in => [], :out => []))


# affix-handler

dicCODE["is it ست?"] =
    Functor(d -> d["state"] = d["affix"] == "ست" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ی?"] = 
    Functor(d -> d["state"] = d["affix"] == "ی" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ات?"] = 
    Functor(d -> d["state"] = d["affix"] == "ات" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ان?"] = 
    Functor(d -> d["state"] = d["affix"] == "ان" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ش?"] =
    Functor(d -> d["state"] = d["affix"] == "ش" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it م?"] = 
    Functor(d -> d["state"] = d["affix"] == "م" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))


dicCODE["is it مان?"] =
    Functor(d -> d["state"] = d["affix"] == "مان" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it می?"] =
    Functor(d -> d["state"] = d["affix"] == "می" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ون?"] =
    Functor(d -> d["state"] = d["affix"] == "ون" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ید?"] =
    Functor(d -> d["state"] = d["affix"] == "ید" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it یم?"] =
    Functor(d -> d["state"] = d["affix"] == "یم" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ن?"] =
    Functor(d -> d["state"] = d["affix"] == "ن" ? "yes" : "no",
            Dict(:in => ["affix"], :out => ["state"]))



dicCODE["return \"st\""] = 
    Functor(d -> d["res"] = "st",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"ast\""] =
    Functor(d -> d["res"] = "ast",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"i\""] =
    Functor(d -> d["res"] = "i",
            Dict(:in => [], :out => ["res"]))
            
dicCODE["return \"ye\""] = 
    Functor(d -> d["res"] = "ye",
            Dict(:in => [], :out => ["res"]))
          
dicCODE["return \"at\""] =
    Functor(d -> d["res"] = "at",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"'at\""] =
    Functor(d -> d["res"] = "'at",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yan\""] =
        Functor(d -> d["res"] = "yan",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"an\""] =
    Functor(d -> d["res"] = "an",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"as\""] =
    Functor(d -> d["res"] = "as",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"es\""] = 
    Functor(d -> d["res"] = "es",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"om\""] =
    Functor(d -> d["res"] = "om",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"am\""] = 
    Functor(d -> d["res"] = "am",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"man\""] = 
    Functor(d -> d["res"] = "man",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"na\""] =
    Functor(d -> d["res"] = "na",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"eman\""] = 
    Functor(d -> d["res"] = "eman",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"omi\""] = 
    Functor(d -> d["res"] = "omi",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"mi\""] = 
    Functor(d -> d["res"] = "mi",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yun\""] = 
    Functor(d -> d["res"] = "yun",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"un\""] =
        Functor(d -> d["res"] = "un",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yad\""] = 
    Functor(d -> d["res"] = "yad",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"im\""] = 
    Functor(d -> d["res"] = "im",
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yam\""] = 
    Functor(d -> d["res"] = "yam",
            Dict(:in => [], :out => ["res"]))


# dicCODE["return \"im\""] =
# dicCODE["return \"im\""] =
# dicCODE["return \"an\""] =

dicCODE["is it a suffix?"] =
    Functor(d -> (aff = d["affix"]; d["state"] = d["word"][length(aff)] == aff ? "yes" : "no"),
            Dict(:in => ["word", "affix"], :out => ["state"]))

dicCODE["is there only one instance of the affix?"] = 
    Functor(d -> (d["state"] = py"""has_only_one_search_pos"""(d["data"]); d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["use it! "] =
    Functor(d -> d,
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["is the word before it a verb?"] =
    Functor(d -> d,
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["is the word to-which it's attached, a noun?"] =
    Functor(d -> (d["state"] = d["pos"] == "Noun" ? "true" : "false"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["is the word to-which it's attached, a number or چند?"] = 
    Functor(d -> (d["state"] = contains(d["word"], "چند") ? "yes" :
        d["pos"] == "Number" ? "true" : "false"; d),
            Dict(:in => ["pos", "affix"], :out => ["state"]))

dicCODE["is the verb root to-which it's attached, marked as v2 in the database?"] = 
    Functor(d -> d,
            Dict(:in => ["verb_root"], :out => ["state"]))

dicCODE["does the verb root to-which it's attached, end in any of the /e, a, u/ sounds?"] =
    Functor(d -> d,
            Dict(:in => ["verb_root"], :out => ["state"]))

dicCODE["is there a space or semi-space before it?"] =
    Functor(d -> (n = first(findlast(d["affix"], d["word"]));
                  if n > 1
                      d["word"][n-1:n-1] == " " ? 
                           d["state"] = "yes" :
                        if n > 4
                            d["state"] = 
                                d["word"][n-3:n-1+length(d["affix"])] == string('\u200c', d["affix"]) ?
                                "yes" : "no"
                        else
                            d["state"] = "no"
                        end
                  else
                      d["state"] = "no" 
                  end; 
                  d),
            Dict(:in => ["word", "affix"], :out => ["state"]))

dicCODE["is is found in affixes?"] =
    Functor(d -> (d["data"] = py"""affix_search"""(d["affix"]);
                  d["state"] = length(d["data"]) > 0 ? "true" : "false"; d),
            Dict(:in => ["affix"], :out => ["state", "data"]))

dicCODE["return its transliteration in affixes"] =
    Functor(d -> (d["res"] = d["data"][1]["PhonologicalForm"];
                  d["state"] = length(d["data"]) > 0 ? "true" : "false";
                  d),
            Dict(:in => ["data"], :out => ["res"]))


dicCODE["is the prefix ب or بی?"] =
    Functor(d -> (d["state"] = d["prefix"] in ["ب" ,"بی"] ? "true" : "false";
                  d),
            Dict(:in => ["prefix"], :out => ["state"]))
    

#dicCODE["do both verb roots exist in the verb?"] =
#    Functor(d -> (#d["res"] = d["data"][1]["PhonologicalForm"];
#                  d["state"] = length(d["data"]) > 0 ? "true" : "false";
#                  d),
#            #Dict(:in => ["stem"], :out => ["state"])
#            Dict(:in => ["word", "lemma"], :out => ["state"]))

dicCODE["do both verb roots exist in the verb?"] =
    Functor(d -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 2 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))


dicCODE["use the second verb root!"] =
    Functor(d -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["use the first verb root!"] =
    Functor(d -> (d["state"] = split(d["lemma"], "#")[1]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["undo the change to the first verb root and use it!"] =
    Functor(d -> (d["res"] = split(d["lemma"], "#")[1]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["undo the change to the second verb root and use it!"] =
    Functor(d -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["is there an آ in the verb roots?"] = 
    Functor(d -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

#===
dicCODE["change the first آ in the verb root(s) to ا."] =
    Functor(d -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["undo the change to the verb root and use it! "] = 
    Functor(d -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))
===#
