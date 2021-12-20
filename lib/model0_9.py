
import lib.get_assets as assets


d_corrects = {'ي' : 'ی',
              'ك'   : 'ک'}

d_corrects = dict([(d[0],d[1]) for d in d_corrects.items()])


def has_farsi(text):
    if len([i for i in range(len(text)) \
        if text[i] in 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی']) == 0:
        return False
    else:
        return True


def normalise(text):
    text = ''.join([d_corrects.get(s, s) for s in list(text)])
    return assets.normalizer.normalize(text)


def votation_entries(l_search, entries=True):

    d_results = {}
    for item in l_search:
        form = item['PhonologicalForm']
        if d_results.get(form, False):
            d_results[form] += item['Freq'] if entries else 1
        else:
            d_results[form] = item['Freq'] if entries else 1

    M = max(d_results.values())
    idx = list(d_results.values()).index(M)
    return list(d_results.items())[idx][0]


def filter_search(l_search, pos_pos=None, pos_neg=None):

    if not pos_pos is None: # filter pos_pos only
        l_search_tmp = [d for d in l_search
                        if assets.d_map_FLEXI.get(d['SynCatCode'], False)==pos_pos]
        if len(l_search_tmp) > 0:
            l_search = l_search_tmp

    if not pos_neg is None: # filter pos_neg out
        l_search = [d for d in l_search
                    if assets.d_map_FLEXI.get(d['SynCatCode'], False)!=pos_neg]

    return l_search


def general_search(wrd, choice_model=votation_entries, pos_pos=None, pos_neg=None):

    # all matches
    l_search = assets.df_Entries[assets.df_Entries['WrittenForm']==wrd].to_dict('records')
    l_search = filter_search(l_search, pos_pos, pos_neg)

    if len(l_search) == 0: # if not found, returns wrd
        return wrd
    if len(l_search) == 1: # if one unique found, return the form
        return l_search[0]['PhonologicalForm']
    else:
        return choice_model(l_search)


def get_affixes(w_original, w_transformed):
    """
        function getting prefix and suffix
        args:
            w_original:     original string
            w_transformed:  stem or lemma
        output:
            dic:    {'prefix': '...', 'suffix': '...'}
    """
    pre_idx = 0
    suf_idx = len(w_original)
    for i in range(len(w_original)-len(w_transformed)+1):
        if w_original[i:i+len(w_transformed)]==w_transformed:
            pre_idx = i
        if w_original[len(w_original)-1-len(w_transformed)-i:len(w_original)-1-i]==w_transformed:
            suf_idx = suf_idx - 1 - i

    prefix = w_original[:pre_idx]
    suffix = w_original[suf_idx:]

    return {'prefix': prefix, 'suffix': suffix}


def affix_search(affix, pos_pos=None, pos_neg=None):

    if affix == '':
        return ''

    l_search = assets.df_Affixes[assets.df_Affixes['Affix']==affix].to_dict('records')
    l_search = filter_search(l_search, pos_pos, pos_neg)

    if len(l_search) == 0: # if not found, returns affix
        return affix
    elif len(l_search) == 1:
        return l_search[0]['PhonologicalForm']
    else:
        return votation_entries(l_search, entries=False)


def search_stem(wrd, pos_pos=None, pos_neg=None):
    """
        Simple procedure extracting a stem from the string start,
        just searching for the longest substring the entries_DB
    """
    l_search = None
    for i in range(len(wrd), 0, -1):
        if assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            break

    if not l_search is None:
        stem = votation_entries(l_search)
        suffix = recu_affixes(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
        return stem + suffix
    else:
        return wrd


def process_noun(wrd):
    """
        Process nouns
    """
    pos = 'Noun'
    w = general_search(wrd, pos_pos=pos)
    if w != wrd:
        return w

    stem = assets.stemmer.stem(wrd)
    d_affixes = get_affixes(wrd, stem)
    prefix = affix_search(d_affixes['prefix'])
    suffix = affix_search(d_affixes['suffix'])
    stem = general_search(stem, pos_pos=pos)

    w = prefix + stem + suffix
    if not has_farsi(w):
        return w
    else:
        # fall back if above procedure failed
        # and returned farsi characters
        return search_stem(wrd, pos_pos=pos)


def process_verb(verb):
    """
        Process verb
    """
    pos = 'Verb'
    l_verbs = verb.split('_')
    l_wrd = []
    for wrd in l_verbs:
        wrd_2 = general_search(wrd, pos_pos=pos)
        lemma = assets.lemmatizer.lemmatize(wrd)
        if wrd_2 != wrd:
            wrd = wrd_2
        elif lemma == wrd:
            wrd = general_search(wrd, pos_pos=pos)
        else:
            l_stem = [w for w in lemma.split('#') \
                      if  w.replace('آ',
                                    'ا') in wrd]
            l_stem = [w for w in l_stem if w != '']
            if len(l_stem) == 2:
                if 'آ' in l_stem[0]:
                    d_affixes = get_affixes(wrd.replace('ا',
                                                        'آ'), l_stem[0])
                else:
                    d_affixes = get_affixes(wrd, l_stem[0])

                if d_affixes['prefix'] in ['ب', 'بی']:
                    stem = l_stem[1] # if len(stem[0]) < len(stem[1]) else stem[1]
                else:
                    stem = l_stem[0]
            elif len(l_stem) == 1:
                stem = l_stem[0]
            else:
                stem = lemma.split('#')[-1]

            if 'آ' in stem:
                d_affixes = get_affixes(wrd.replace('ا',
                                                    'آ'), stem)
            else:
                d_affixes = get_affixes(wrd, stem)

            stem = general_search(stem, pos_pos=pos)

            prefix = recu_affixes(d_affixes['prefix'], pos_pos=pos)
            suffix = recu_affixes(d_affixes['suffix'], pos_pos=pos)
            wrd = prefix + stem + suffix

        l_wrd.append(wrd)

    w = ' '.join(l_wrd)

    if not has_farsi(w):
        return w
    else:
        # fall back if above procedure failed
        # and returned farsi characters
        return search_stem(wrd, pos_pos=pos)


def recu_entries(wrd, pos_pos=None, pos_neg=None):
    """
        Recursive search in entries_DB:
        decompose wrd into largest substrings found in DB.
    """
    for i in range(len(wrd), 0, -1):
        if assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            return votation_entries(l_search) + recu_entries(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
            break

    return wrd


def recu_affixes(wrd, pos_pos=None, pos_neg=None):
    """
        Recursive search in affixes_DB:
        decompose wrd into largest substrings found in DB.
    """
    for i in range(len(wrd), 0, -1):
        if assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            return votation_entries(l_search, entries=False) + recu_affixes(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
            break

    return wrd
