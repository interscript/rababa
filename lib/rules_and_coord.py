
import lib.get_assets as assets
import lib.model0_9 as lib0_9


"""
    Semi Space Rules
"""


def semispace_9(wrd, pos=None):
    if len(wrd) > 2:
        if wrd[:-3] == '\u200cات':
            return process_wrd(wrd[:-3], pos, state=0) + "'at"
    return wrd


def semispace_19(wrd, pos=None):
    if not "\u200c" in wrd:
        return wrd
    else:
        l_wrd = wrd.split("\u200c")
        _str = []
        for i in range(len(l_wrd)):
            _str.append(process_wrd(l_wrd[i], pos, state=1))

    return "".join(_str)


"""
    Suffix Rules
"""


def suffix_13(wrd, pos=None):
    if len(wrd) > 3:
        if wrd[-3:] == 'مان':
            w = process_wrd(wrd[:-3], pos, state=2) # 'Verb'
            w += "mAn" \
                if w[-1] in ["a", "e", "o", "A", "i", "u"] else "emAn"
            return w
    return wrd


def suffix_7(wrd, pos=None):
    if not "ی" == wrd[-1]:
        return wrd
    else:
        w = process_wrd(wrd, pos, state=2)

        if w[-1] == 'i':
            w = w[:-1]
        elif w[-2:] == 'ye':
            w = w[:-2]

        if pos == 'Verb':
            w = w + 'i'
        else:
            if w[-1] in ['a','e','o','A','u']:
                w = w + 'ye'
            else:
                w = w + 'i'
        return w


def suffix_8(wrd, pos=None):
    if not wrd[-1] in ["ٔ", "ۀ"]:
        return wrd
    else:
        wrd = process_wrd(wrd[:-1], pos, state=2)
        if wrd[-1] == "e":
            wrd += "ye"
        else:
            wrd += "eye"
        return wrd


def suffix_9(wrd, pos=None):
    if not wrd[-2:] == "ات":
        return wrd
    else:
        d_affixes = lib0_9.get_affixes(wrd, "ات")
        return process_wrd(wrd[:-2], pos, state=2) + 'At'


def suffix_10(wrd, pos=None):
    if not wrd[-2:]=="ان":  # pos == 'Noun'
        return wrd
    else:
        if pos == 'Noun':
            w = lib0_9.process_noun(wrd)
            if not lib0_9.has_farsi(w):
                if wrd[-3] == "ی" and wrd[-3:] != 'yAn':
                    return w[:-2] + 'yAn'
                else:
                    return w
        elif len(wrd) == 2:
            return "An"
        elif wrd[-3] == "ی":
            return process_wrd(wrd[:-2], pos, state=2) + "yAn"
        else:
            return process_wrd(wrd[:-2], pos, state=2) + "An"


def suffix_11(wrd, pos=None):
    if not "ش" == wrd[-1]:
        return wrd
    else:
        if pos == "Noun":
            return lib0_9.process_noun(wrd)[:-2] + "aS"
        elif pos == "Verb":
            return lib0_9.process_verb(wrd)[:-2] + "eS"
        else:
            return process_wrd(wrd[:-1], pos, state=2) + "eS"


def suffix_12(wrd, pos=None):
    if not 'م' == wrd[-1]:
        return wrd
    else:
        if pos == 'Number':
            return process_wrd(wrd[:-1], pos, state=2) + 'om'
        else:
            if pos == 'Noun':
                w = lib0_9.process_noun(wrd)
                if not lib0_9.has_farsi(w):
                    return w
                else:
                    return lib0_9.process_noun(wrd[:-1]) + 'am'
            elif pos == 'Verb':
                w = lib0_9.process_verb(wrd)
                if not lib0_9.has_farsi(w):
                    return w
                else:
                    return lib0_9.process_verb(wrd[:-1]) + 'am'
            else:
                return process_wrd(wrd[:-1], pos, state=2) + 'am'


def suffix_14(wrd, pos=None):
    if not "می" == wrd[-2:]:
        return wrd
    else:
        return process_wrd(wrd[2:], pos, state=2) + "omi"


def suffix_16(wrd, pos=None):
    if not "ون" == wrd[-2:]:
        return wrd
    else:
        w = process_wrd(wrd, pos, state=2)
        if w != wrd:
            if w[-3:] == 'iun':
                w = w[:-2] + 'yun'
            return w
        else:
            w = process_wrd(wrd[:-2], pos, state=2)
            w += "yun" if w[-1] == "i" else "un"
            return w


def suffix_17(wrd, pos=None):
    if not wrd[-2:] == 'ید':
        return wrd
    else:
        if pos == "Verb":
            w = lib0_9.process_verb(wrd)
            if w[-3] in ['e', 'A', 'u']:
                w = w[:-2] + 'yad'
            return w
        else:
            w = process_wrd(wrd[:-2], pos, state=2)
    return wrd


def suffix_18(wrd, pos=None):
    if not 'یم' == wrd[-2:]:
        return wrd
    else:
        if pos == 'Verb':
            w = process_wrd(wrd[:-2], pos, state=2)
            if w[-1] in ['e', 'A', 'u']:
                w = w + 'yam'
            else:
                w = w + 'im'
        else:
            w = process_wrd(wrd, pos, state=2)
        return w


def suffix_21(wrd, pos=None):
    if not wrd[-1] == "ن":
        return wrd
    else:
        return process_wrd(wrd[:-1], 'Verb', state=2) + 'an'


"""
    Prefixes Rules
"""


def prefix_14(wrd, pos=None):
    if not 'می' == wrd[:2]:
        return wrd
    else:
        return "mi" + process_wrd(wrd[2:], pos, state=3)


def prefix_21(wrd, pos=None):
    #print(wrd)
    #if wrd == '':
    #    print('bugfix')

    if not wrd[0] == "ن":
        return wrd
    else:
        print('prefix_21:: ', wrd)
        return 'na' + process_wrd(wrd[1:], pos=pos, state=3)


def prefix_22(wrd, pos=None):
    if not wrd[:2] in ["بی", "نی"]:
        return wrd
    else:
        w = process_wrd(wrd, pos=pos, state=3)
        if not w[:3] in ['nay', 'biy']:
            if w[:2] == 'bi':
                w = 'biy' + w[2:]
            elif w[:2] == 'na':
                w = 'nay' + w[2:]
            else:
                w = lib0_9.affix_search(wrd[:2]) + \
                        process_wrd(wrd[2:], pos, state=3)
        if len(w) > 3:
            if w[3] == "'":
                w = w[:3] + w[4:]
        return w

"""
    Root Rules
"""

def root_23(wrd, pos=None):

    if not 'رو' in wrd:
        return wrd
    else:

        d_affixes = lib0_9.get_affixes(wrd, 'رو')
        stem = 'ro' if d_affixes['suffix'] == '' else 'rav'
        prefix = lib0_9.recu_affixes(d_affixes['prefix'], pos_pos=pos)
        suffix = lib0_9.recu_affixes(d_affixes['suffix'], pos_pos=pos)
        return prefix + stem + suffix


def process_wrd(wrd, pos, state=0):

    if wrd == '':
        return ''

    w = lib0_9.general_search(wrd, pos_pos=pos)
    if w != wrd:
        return w

    if state == 0:
        # Process semispaces
        w = semispace_9(wrd, pos=None) # '\u200cات'
        if w != wrd:
            return w
        w = semispace_19(wrd, pos) # recursion "\u200c"
        if w != wrd:
            return w

        w = root_23(wrd, pos) # root 'رو' in wrd
        if w != wrd:
            return w

        state = 1

    # Process prefixes
    if state == 1:

        w = suffix_13(wrd, pos) # مان
        if not lib0_9.has_farsi(w):
            return w
        w = suffix_9(wrd, pos) # ات
        w = suffix_10(wrd, pos) # ان
        w = suffix_16(wrd, pos) # ون
        w = suffix_14(wrd, pos) # می
        w = suffix_17(wrd, pos) # ید
        w = suffix_18(wrd, pos) # یم
        if not lib0_9.has_farsi(w):
            return w
        w = suffix_7(wrd, pos) # ی
        w = suffix_8(wrd, pos) # ۀ
        w = suffix_11(wrd, pos) # ش
        w = suffix_12(wrd, pos) # م
        w = suffix_21(wrd, pos) # ن
        if not lib0_9.has_farsi(w):
            return w
        state = 2

    if state == 2:

        w = prefix_22(wrd, pos) # بی and نی
        w = prefix_14(wrd, pos) # می
        w = prefix_21(wrd, pos) # ن
        if not lib0_9.has_farsi(w):
            return w
        state = 3
    #
    if state == 3:
        if pos == "Noun":
            w = lib0_9.process_noun(wrd)
        elif pos == "Verb":
            w = lib0_9.process_verb(wrd)
        else:  # general case
            #w = lib0_9.general_search(wrd, pos_pos=pos)
            w = lib0_9.search_stem(wrd, pos_pos=pos)

    if not lib0_9.has_farsi(w):
        return w
    else:
        # fall back if above procedure failed
        # and returned farsi characters
        return lib0_9.search_stem(wrd, pos_pos=pos)


def post_process(txt):
    txt_out = [0]
    for t in txt:
        if t != txt_out[-1]:
           txt_out.append(t)
    return ''.join(txt_out[1:])


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

        l_transcribed.append(
            process_wrd(wrd, pos))

    # return post_process(
    #         " ".join(l_transcribed))
    return " ".join(l_transcribed)
