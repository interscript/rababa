

include("hazm/py_code.jl")


dicCODE = Dict{String, Functor}()


# transliterator

dicCODE["change all instances of ي and ك in the text to ی and ک"] =
    Functor((d,e=nothing,f=nothing) -> (d["word"]=py"""normalise"""(d["word"]); d),
            Dict(:in => ["word"], :out => ["word"]))

dicCODE["is the word found in the db?"] =
    Functor((d,e=nothing,f=nothing) -> (d["data"]=py"""search_db"""(d["word"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["word", "pos"], :out => ["data", "state"]))

dicCODE["is it a verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["pos"] == "Verb" ? "yes" : "no"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["lemmatize it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = lemmatizer.lemmatize(d["word"]); d),
            Dict(:in => ["word"], :out => ["lemma"]))

dicCODE["includes underscores?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = contains(d["lemma"], "_") ? "yes" : "no"; d),
            Dict(:in => ["lemma"], :out => ["state"]))

dicCODE["does only one of the verb roots exist in the verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 1 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))

dicCODE["output it!"] =
    Functor((d,e=nothing,f=nothing) -> (haskey(d, "res") ? d :
                   typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                        (d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]); d) :
                         d),
            Dict(:in => ["data"], :out => []))

dicCODE["collision?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(d["data"]) == 1 ? "no" : "yes"; d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["output its transliteration!"] =
    Functor((d,e=nothing,f=nothing) -> (
            haskey(d, "res") ? d :
                typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                    (v = py"""return_highest_search_pos"""(d["data"], d["pos"]);
                     d["res"] = v[1]; d["SynCatCode"] = v[2]; d) :
                    (d["res"] = d["word"]; d)),
            Dict(:in => [], :out => []))

dicCODE["return its transliteration!"] = dicCODE["output its transliteration!"]


dicCODE["stem it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["lemma"] = stemmer.stem(d["word"]); d),
            Dict(:in => ["word"], :out => ["lemma"])) # lemma

dicCODE["is the verb root found in the db?"] =
    Functor((d,e=nothing,f=nothing) -> (d["data"]=py"""search_db"""(d["lemma"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["lemma", "pos"], :out => ["data", "state"]))

dicCODE["does the root of the word exist in the database?"] =
    Functor((d,e=nothing,f=nothing) -> (d["data"]=py"""search_db"""(d["lemma"], d["pos"]);
            d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["lemma", "pos"], :out => ["data", "state"]))

dicCODE["transliterate each side of underscore separately in proper order"] =
    Functor((d,e=nothing,f=nothing) -> split(d["lemma"], "_") |>
                (D -> map(x -> py"""return_highest_search_pos"""(x, d["pos"]), D) |> join),
            Dict(:in => ["lemma"], :out => ["res"]))


# collision-handler

dicCODE["is there an instance of the word with the desired pos?"] =
    Functor((d,e=nothing,f=nothing) -> (
            d["state"] = py"""has_entries_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["is there only one instance of the word with the desired pos?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = py"""has_only_one_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["return the transliteration of the instance with the desired pos!"] =
    Functor((d,e=nothing,f=nothing) ->
        ((d["res"], d["SynCatCode"]) = py"""return_highest_search_pos"""(d["data"], d["pos"]);
         d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the desired pos that has the highest frequency!"] =
    Functor((d,e=nothing,f=nothing) ->
        (v = py"""return_highest_search_pos"""(d["data"], d["pos"]);
         d["res"] = v[1]; d["SynCatCode"] = v[2]; d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the highest frequency!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = py"""return_highest_search"""(d["data"]); d),
            Dict(:in => ["data"], :out => ["res"]))


# Full Model

dicCODE["return \"id\""] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "id"; d),
            Dict(:in => [], :out => []))

dicCODE["terminator"] =
    Functor((d,e=nothing,f=nothing) -> d,
            Dict(:in => [], :out => []))


# affix-handler

dicCODE["is it ست?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ست" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ی?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ی" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ات?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ات" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ان?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ان" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ش?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ش" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it م?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "م" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))


dicCODE["is it مان?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "مان" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it می?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "می" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ون?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ون" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ید?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ید" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it یم?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "یم" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ن?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ن" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it بی or نی?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] in ["بی", "نی"] ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["return \"st\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "st"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"ast\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "ast"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"i\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "i"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"ye\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "ye"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"at\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "at"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"'at\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "'at"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yan\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yan"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"an\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "an"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"as\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "as"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"es\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "es"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"om\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "om"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"am\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "am"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"man\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "man"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"na\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "na"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"eman\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "eman"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"omi\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "omi"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"mi\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "mi"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yun\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yun"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"un\""] =
        Functor((d,e=nothing,f=nothing) -> (d["res"] = "un"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yad\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yad"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"im\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "im"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yam\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yam"; d),
            Dict(:in => [], :out => ["res"]))


dicCODE["is it a suffix?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if haskey(d,"suffix")
                d["suffix"] == d["affix"] ? "yes" : "no"
            else
                "no"
            end; d),
        #(aff = d["affix"];
         #d["state"] = d["word"][length(aff)] == aff ? "yes" : "no";
         #d),
            Dict(:in => ["word", "affix"], :out => ["state"]))

dicCODE["is there only one instance of the affix?"] =
    Functor((d,e=nothing,f=nothing) -> (
    d["state"] = py"""has_only_one_search_pos"""(d["data"]); d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["use it! "] =
    Functor((d,e=nothing,f=nothing) ->
            (d["lemma"] = filter(x -> contains(d["word"], x),
                                 split(d["lemma"], "#"))[1];
             d),
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["use it!"] = dicCODE["use it! "]

dicCODE["is the word before it a verb?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["pre_pos"] == "Verb" ? "yes" : "no"; d),
            Dict(:in => ["pre_pos"], :out => ["state"]))

dicCODE["is the word to-which it's attached, a noun?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["pos"] == "Noun" ? "true" : "false"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["is the word to-which it's attached, a number or چند?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = contains(d["word"], "چند") ? "yes" :
        d["pos"] == "Number" ? "true" : "false"; d),
            Dict(:in => ["pos", "affix"], :out => ["state"]))

dicCODE["is the verb root to-which it's attached, marked as v2 in the database?"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = d["SynCatCode"]=="V2" ? "yes" : "no"; d),
            Dict(:in => ["SynCatCode"], :out => ["state"]))

dicCODE["does the verb root to-which it's attached, end in any of the /e, a, u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) -> d,
            Dict(:in => ["verb_root"], :out => ["state"]))

dicCODE["is there a space or semi-space before it?"] =
    Functor((d,e=nothing,f=nothing) -> (n = first(findlast(d["affix"], d["word"]));
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
    Functor((d,e=nothing,f=nothing) -> (d["data"] = py"""affix_search"""(d["affix"]);
                              d["state"] = length(d["data"]) > 0 ? "true" : "false"; d),
            Dict(:in => ["affix"], :out => ["state", "data"]))

dicCODE["return its transliteration in affixes"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["data"][1]["PhonologicalForm"];
         d["state"] = length(d["data"]) > 0 ? "true" : "false";
         d),
            Dict(:in => ["data"], :out => ["res"]))


dicCODE["is the prefix ب or بی?"] =
    Functor((d,e=nothing,f=nothing) ->
                (wrd = d["word"];
                 root = filter(x -> contains(wrd, x), split(d["lemma"], "#"))[1];
                 idx = findfirst(root, wrd)[1];
                 d["state"] = wrd[1:idx] in ["ب", "بی"] ? "yes" : "no"; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["do both verb roots exist in the verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 2 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))


dicCODE["use the second verb root!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["use the first verb root!"] =
    Functor((d,e=nothing,f=nothing) ->
            (d["lemma"] = split(d["lemma"], "#")[1]; d),
            Dict(:in => ["lemma"], :out => ["res"]))


dicCODE["is there an آ in the verb roots?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = contains(d["lemma"], "آ") ?  "yes" : "no"; d),
            Dict(:in => ["lemma"], :out => ["state"]))


dicCODE["change the first آ in the verb root(s) to ا."] =
    Functor((d,e=nothing,f=nothing) -> (d["lemma"] = split(d["lemma"], "#") |>
                    (Ls -> map(x -> (idx = findfirst("آ", x) |> first;
                                     string(replace(x[1:idx], "آ" => "ا"), x[idx+2:end])),
                               Ls) |>
                            (L -> join(split(L, "a"), "#"))); d),
            Dict(:in => ["lemma"], :out => ["lemma"]))


dicCODE["undo the change to the first verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = replace(split(d["lemma"], "#")[1], "آ" => "ا"); d),
            Dict(:in => ["lemma"], :out => ["res"]))

dicCODE["undo the change to the second verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = replace(split(d["lemma"], "#")[2], "آ" => "ا"); d),
            Dict(:in => ["lemma"], :out => ["res"]))


dicCODE["does the transliteration of the segment before it end in any of the /a, i, u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["l_res"][end][end-1:end] in ["a", "e", "i"] ? "yes" : "no"; d),
            Dict(:in => ["l_res"], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in any of the /a,e,o,a,u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["l_res"][end][end-1:end] in ["a", "e", "o","u"] ? "yes" : "no"; d),
            Dict(:in => ["l_res"], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in /i/?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["l_res"][end][end-1:end] == "i" ? "yes" : "no"; d),
            Dict(:in => ["l_res"], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in any of the /a,e,o,a,i,u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["l_res"][end][end-1:end] in ["a", "e", "o","u", "i"] ? "yes" : "no"; d),
            Dict(:in => ["l_res"], :out => ["state"]))


dicCODE["is there anything after the word root?"] =
    Functor((d,e=nothing,f=nothing) ->
    (d["state"] = length(d["word"]) < last(findlast(reverse(d["lemma"]), reverse(d["word"]))) ? "yes" : "no"; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["is there anything before the word root?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = 1 == first(findfirst(d["lemma"], d["word"])) ? "no" : "yes"; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["is it a single letter?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(d["word"]) == 1 ? "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))


dicCODE["is it found in affixes?"] =
    Functor((d,e=nothing,f=nothing) ->
        (data = py"""affix_search"""(
            haskey(d, "affix") ? d["affix"] : d["word"]);
         d["state"] = if typeof(data) == String
                "no"
            else
                length(data) > 0 ? "yes" : "no"
         end;
         d),
            Dict(:in => [], :out => ["state"]))


dicCODE["return the transliteration with t as its pos"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = py""""search_db"""(d["affix"], pos); d),
            Dict(:in => ["affix"], :out => ["res"]))


dicCODE["return its transliteration then omit the ' symbol in the beginning of the word root that comes after it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = string(collect(d["data"][:PhonologicalForm])[1], "-'"); d),
            Dict(:in => ["data"], :out => ["res"]))


dicCODE["is the word root, رو recognized as a verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["pos"] == "Verb" ? "yes" : "no";
                  d["res"] = "rav";d),
            Dict(:in => ["data", "pos"], :out => ["state", "res"]))


dicCODE["change the word root's transliteration from /rav/ to /ro/"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "ro"; d),
            Dict(:in => ["res"], :out => ["res"]))


dicCODE["mark it as prefix"] =
    Functor((d,e=nothing,f=nothing) ->
        (l = length(collect(d["lemma"]));
         d["prefix"] = join(collect(d["word"])[1:l-1]); #join(collect(d["word"])[1:end-l]);
         d["affix"] = d["prefix"];
         d["res_root"] = d["res"]; d),
            Dict(:in => ["word", "lemma"], :out => ["prefix"]))


dicCODE["mark it as suffix"] =
    Functor((d,e=nothing,f=nothing) ->
      (d["suffix"] = (id = last(findlast(d["lemma"], d["word"]));
       try
           d["word"][id+1:end]
       catch
           d["word"][id+2:end]
       end);
       d["res_root"] = d["res"];
       delete!(d, "res");
       d["affix"] = d["suffix"];
       #d["word"] = d["suffix"]; jair
       d["data"] = py"""affix_search"""(d["affix"]);
       d["brain"] = "fixbug";
       d),
            Dict(:in => ["word", "lemma"], :out => ["suffix"]))


dicCODE["add it to the beginning of the root's transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = haskey(d, "res_prefix") ?
            string(d["res_prefix"], d["res_root"]) :
            string(d["res"], d["res_root"]); d),
            Dict(:in => ["res_root"], :out => ["res"]))


dicCODE["add it to the end of the root's transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = haskey(d, "res_suffix") ?
            string(d["res_root"], d["res_suffix"]) :
            string(d["res_root"], d["res"]); d),
            Dict(:in => ["res_root", "res"], :out => ["res"]))


dicCODE["undo the change to the verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = d["lemma"] |>
                    (x -> (idx = findfirst("ا", x) |> first;
                           string(replace(x[1:idx], "آ" <= "ا"),
                                  x[idx+2:end])));
                  d),
            Dict(:in => ["lemma"], :out => ["res"]))


#dicCODE["undo the change to the verb root and use it! "] = dicCODE["undo the change to the verb root and use it!"]


dicCODE["return the concatenation of all the returned transliterations."] =
    Functor((d,e=nothing,f=nothing) ->
        (#d["res"] = if haskey(d, "suffix")
        #    string(d["res_root"], d["res"])
        #else
        #    string(d["res"], d["res_root"])
        #end;
        d),
            Dict(:in => [], :out => ["res"]))


dicCODE["transliterate it using affix-handler"] =
    Functor((d,e=nothing,f=nothing) ->
        (brainName = d["brain"];
         d["res"] = if haskey(d, "prefix")
                        (interfaceName = "affix-handler";
                         node = e[interfaceName];
                         d["affix"]=d["prefix"];
                         if haskey(d, "res")
                             d["res_root"] = d["res"]
                             delete!(d, "res")
                         end;
                         d["data"] = py"""affix_search"""(d["affix"]);
                         d["res_prefix"] = runAgent(node, e, f, d); d)
                    elseif haskey(d, "suffix")
                        (interfaceName = "affix-handler";
                         node = e[interfaceName];
                         d["affix"] = d["suffix"];
                            if haskey(d, "res")
                                d["res_root"] = d["res"]
                                delete!(d, "res")
                            end;
                         d["data"] = py"""affix_search"""(d["affix"]);
                         d["res_suffix"] = runAgent(node, e, f, d); d)
                    end;
            d["brain"] = brainName; d),
            Dict(:in => [], :out => ["res"]))


dicCODE["run affix-handler on affix vector"] =
    Functor((d,e=nothing,f=nothing) ->
        (interfaceName = "affix-handler";
         res = map(w -> (node = e[interfaceName];
                        if haskey(d, "res")
                            delete!(d, "res")
                        end;
                        d["affix"] = w;
                        d["data"] = py"""affix_search"""(w);
                        runAgent(node, e, f, d)),
                   d["l_affix"]);
         d["res"] = join(res, ""); d),
            Dict(:in => ["l_affix"], :out => ["res"]))


dicCODE["find the longest substring of the input that exists in the database."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["d_substring"] = py"""largest_root_and_affixes"""(d["word"]);
         # d["res"] = join(py"""recu_entries"""(d["word"]), "");
        d),
            Dict(:in => ["word"], :out => ["res"]))


dicCODE["transliterate each side of it separately in proper order and put its transliteration with the highest frequency between them."] =
    Functor((d,e=nothing,f=nothing) ->
        (# root
         res_root = py"""recu_entries"""(d["d_substring"]["root"], d["pos"]) |>
            (D -> typeof(D) == String ? D : D[1]);
         # prefix
         interfaceName = "affix-handler";
         node = e[interfaceName];
         prefix = d["d_substring"]["prefix"];
         d["affix"] = prefix;
         d["prefix"] = prefix;
         haskey(d, "res") ? delete!(d, "res") : "";
         d["data"] = py"""affix_search"""(d["affix"]);
         res_prefix = runAgent(node, e, f, d);
         # suffix
         node = e[interfaceName];
         suffix = d["d_substring"]["suffix"];
         delete!(d, "prefix");
         d["affix"] = suffix;
         d["suffix"] = suffix;
         haskey(d, "res") ? delete!(d, "res") : "";
         d["data"] = py"""affix_search"""(d["affix"]);
         res_suffix = runAgent(node, e, f, d);
         # res aggregating strings
         d["res"] = string(res_prefix, res_root, res_suffix); d),
            Dict(:in => ["d_substring"], :out => ["res"]))

dicCODE["move the longest substring of the input that exists in affixes and starts in the beginning of the input to affix vector. if the input is not empty and no substring of the input can be found in affixes, move contents of affix vector back to the input then run terminator on it."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["l_affix"] = py"""recu_affixes_subs"""(d["affix"], d["pos"]); d),
            Dict(:in => ["affix"], :out => ["res"]))
