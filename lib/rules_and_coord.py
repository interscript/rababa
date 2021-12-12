import lib.get_assets as assets
import lib.model0_9 as lib0_9


def rule_7(wrd, pos=None, pos_last=None):
    if not "ی" == wrd[-1]:
        return wrd
    else:
        wrd = process_wrd(wrd, pos)
        if not pos_last == "Verb":
            if wrd[-1] == "i":
                wrd = wrd[:-1] + "ye"
        return wrd


def rule_8(wrd, pos=None):
    if not wrd[-1] in ["ٔ", "ۀ"]:
        return wrd
    else:
        wrd = process_wrd(wrd[:-1], pos=pos)
        if wrd[-1] == "e":
            wrd += "ye"
        else:
            wrd += "eye"
        return wrd


def rule_9(wrd, pos=None):
    if not "ات" in wrd:
        return wrd
    else:
        if "ات‌" in wrd:
            d_affixes = lib0_9.get_affixes(wrd, "ات‌")  # , stem)
            stem = "At"
        elif "\u200cات" in wrd:
            d_affixes = lib0_9.get_affixes(wrd, "\u200cات")
            stem = "'at"
        else:
            d_affixes = lib0_9.get_affixes(wrd, "ات")
            stem = "At"

    if len(d_affixes["prefix"]) > len(d_affixes["suffix"]):
        return (
            lib0_9.general_search(d_affixes["prefix"], pos_pos=pos)
            + stem
            + lib0_9.affix_search(d_affixes["suffix"])
        )
    else:
        return (
            lib0_9.affix_search(d_affixes["prefix"])
            + stem
            + lib0_9.general_search(d_affixes["suffix"], pos_pos=pos)
        )


def rule_10(wrd, pos=None):
    if not "ان" in wrd:
        return wrd
    else:
        if len(wrd) == 2:
            return "An"
        elif wrd[3] == "ی":
            return lib0_9.general_search(wrd[:-2], pos_pos=pos) + "yAn"
        else:
            return lib0_9.general_search(wrd[:-2], pos_pos=pos) + "An"


def rule_11(wrd, pos=None):
    if "ش" != wrd[-1]:
        return wrd
    else:
        if pos == "Noun":
            return lib0_9.process_noun(wrd[:-1]) + "aS"
        elif pos == "Verb":
            return lib0_9.process_verb(wrd[:-1]) + "eS"
        else:
            return lib0_9.general_search(wrd[:-1], pos_pos=pos) + "eS"


def rule_13(wrd, pos=None):
    if len(wrd) > 3:
        if wrd[-4:] == "\u200cمان":
            wrd = process_wrd(wrd[:-4], pos)
            wrd += "mAn" if wrd[-1] in ["a", "e", "o", "A", "i", "u"] else "emAn"
            return wrd
    return wrd


def rule_14(wrd, pos=None):
    if not "می" in wrd:
        return wrd
    else:
        if "می‌" == wrd[:3]:
            return "mi" + process_wrd(wrd[3:], pos)
        elif "می" == wrd[-2:]:
            return lib0_9.general_search(wrd[:-2]) + "omi"


def rule_16(wrd, pos=None):
    if not "ون" == wrd[-2:]:
        return wrd
    else:
        w = process_wrd(wrd, pos)
        # print('dbg: ', w)
        if w != wrd:
            return w
        else:
            w = process_wrd(wrd[:-2], pos)
            w += "yun" if w[-1] == "i" else "un"
            return w


def rule_17(wrd, pos=None):
    if not "ید" in wrd:
        return wrd
    elif pos == "Verb":
        l_lemma = [
            w for w in assets.lemmatizer.lemmatize(wrd).split("#") if w == wrd[: len(w)]
        ]
        if len(l_lemma) > 0:
            # print(1)
            return lib0_9.process_verb(wrd[: -len("ید")]) + "id"
        else:
            return lib0_9.process_verb(wrd[: -len("ید")]) + "yad"
    else:
        return wrd


def rule_19(wrd, pos):
    if not "\u200c" in wrd:
        return wrd
    else:
        l_wrd = wrd.split("\u200c")
        M = max([len(w) for w in l_wrd])
        _str = []
        for i in range(len(l_wrd)):
            w = l_wrd[i]
            if len(w) == M:
                _str.append(process_wrd(w, pos))
            else:
                w_tmp = lib0_9.process_verb(w)
                if w != w_tmp:
                    _str.append(w_tmp)
                else:
                    _str.append(lib0_9.recu_affixes(w))
    return "".join(_str)


def rule_21(wrd, pos=None):
    if not (wrd[-1] == "ن" or wrd[0] == "ن"):
        return process_wrd(wrd, pos)
    else:
        if wrd[-1] == "ن":
            return rule_21(wrd[:-1], pos="Verb") + "an"
        elif wrd[0] == "ن":
            return "na" + rule_21(wrd[1:], pos="Verb")


def rule_22(wrd, pos=None):
    if not wrd[:2] in ["بی", "نی"]:
        return wrd
    else:
        suffix = "".join([s for s in lib0_9.recu_affixes(wrd[2:]) if s != "'"])
        return lib0_9.affix_search(wrd[:2]) + suffix


def rule_23(wrd, pos=None):
    if not wrd[-2:] == "رو":
        return wrd
    else:
        l_lemma = assets.lemmatizer.lemmatize(wrd).split("#")
        if len(l_lemma) > 1:
            lemma = [l for l in l_lemma if l == "رو"][0]
            d_affixes = lib0_9.get_affixes(wrd, lemma)
            prefix = d_affixes["prefix"]
            return lib0_9.affix_search(wrd[:-2]) + "ro"
        else:
            d_affixes = lib0_9.get_affixes(wrd, "رو")
            prefix = d_affixes["prefix"]
            return lib0_9.affix_search(wrd[:-2]) + "ro"


def process_wrd(wrd, pos=None, pos_last=None):

    # Run over rules
    wrd = rule_19(wrd, pos=pos)  # \u200c

    #wrd = rule_7(wrd, pos=pos, pos_last=pos_last)
    wrd = rule_8(wrd, pos=pos)
    wrd = rule_9(wrd, pos=pos)
    wrd = rule_10(wrd, pos=pos)
    wrd = rule_11(wrd, pos=pos)
    wrd = rule_13(wrd, pos=pos)
    wrd = rule_14(wrd, pos=pos)
    # wrd = rule_16(wrd, pos=pos)
    wrd = rule_17(wrd, pos=pos)
    # wrd = rule_21(wrd, pos=pos)
    wrd = rule_22(wrd, pos=pos)
    wrd = rule_23(wrd, pos=pos)

    if wrd != wrd:
        return wrd

    if pos == "Noun":
        wrd = lib0_9.process_noun(wrd)
    elif pos == "Verb":
        wrd = lib0_9.process_verb(wrd)
    else:  # general case
        wrd = lib0_9.general_search(wrd, pos_pos=pos)
    return wrd


def run_transcription_0(text):

    l_transcribed = []
    text = lib0_9.normalise(text)
    l_data = [
        (d[0], assets.d_map_HAZM.get(d[1], False))
        for d in assets.tagger.tag(assets.word_tokenize(text))
    ]

    for d in l_data:
        pos = d[1]
        wrd = d[0]
        #if pos == "Noun":
        #    wrd = lib0_9.process_noun(wrd)
        #elif pos == "Verb":
        #    wrd = lib0_9.process_verb(wrd)
        #else:  # general case

        l_transcribed.append(process_wrd(wrd, pos))
    return " ".join(l_transcribed)
