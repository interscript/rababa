
import lib.get_assets as assets


d_corrects = {'ي' : 'ی',
              'ك'   : 'ک'}

d_corrects = dict([(d[0],d[1]) for d in d_corrects.items()])

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
    
    l_search = assets.df_Affixes[assets.df_Affixes['Affix']==affix].to_dict('records')
    l_search = filter_search(l_search, pos_pos, pos_neg)
    
    if len(l_search) == 0: # if not found, returns affix
        return affix
    elif len(l_search) == 1:
        return l_search[0]['PhonologicalForm']
    else:
        return votation_entries(l_search, entries=False)

    
def process_noun(wrd):
    
    stem = assets.stemmer.stem(wrd)
    d_affixes = get_affixes(wrd, stem)
    prefix = d_affixes['prefix']
    suffix = d_affixes['suffix']
    prefix = affix_search(prefix) if prefix != '' else ''
    suffix = affix_search(suffix) if suffix != '' else ''
    stem = general_search(stem, pos_pos='Noun')

    return prefix + stem + suffix


def process_verb(verb):
    
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
            stem=[w for w in lemma.split('#') if w in wrd]
            stem = stem[0] if len(stem) > 0 else lemma.split('#')[-1]
            d_affixes = get_affixes(wrd, stem)
            prefix = d_affixes['prefix']
            suffix = d_affixes['suffix']
        
            stem = general_search(stem, pos_pos=pos)        
            prefix = affix_search(prefix, pos) if prefix != '' else ''
            suffix = affix_search(suffix, pos) if suffix != '' else ''
            wrd = prefix + stem + suffix

        l_wrd.append(wrd)
    
    return ' '.join(l_wrd)


def recu_entries(wrd):
    
    for i in range(len(wrd), 0, -1):
        if assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Entries[assets.df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
            return votation_entries(l_search) + recu_entries(wrd[i:])
            break

    return wrd


def recu_affixes(wrd):
    for i in range(len(wrd), 0, -1):
        if assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].to_dict('records')
            return votation_entries(l_search, entries=False) + recu_affixes(wrd[i:])
            break

    return wrd
