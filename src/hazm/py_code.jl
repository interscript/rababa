

include("get_assets.jl")


py"""

d_corrects = {'ي' : 'ی',
              'ك'   : 'ک'}

d_corrects = dict([(d[0],d[1]) for d in d_corrects.items()])

def normalise(text):
    text = ''.join([d_corrects.get(s, s) for s in list(text)])
    return normalizer.normalize(text)

def search_db(wrd, pos_pos=None):

    # all matches
    l_search = df_Entries[df_Entries['WrittenForm']==wrd].to_dict('records')

    if len(l_search) == 0: # if not found, returns wrd
        return wrd
    else: # if one unique found, return the form
        return l_search

def affix_search(affix, pos_pos=None):

    if affix == '':
        return ''

    l_search = df_Affixes[df_Affixes['Affix']==affix].to_dict('records')

    if len(l_search) == 0: # if not found, returns affix
        return affix
    else:
        return l_search

def has_entries_search_pos(l_search, pos):
    for d in l_search:
        if d_map_FLEXI.get(d['SynCatCode'], False)==pos:
            return "yes"
    return "no"

def has_only_one_search_pos(l_search, pos=None):

    if not pos is None: 
        l_search_tmp = [d for d in l_search
                        if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    else:
        l_search_tmp = l_search

    return "yes" if len(l_search_tmp) == 1 else "no"

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

def return_highest_search_pos(l_search, pos):
    data = [d for d in l_search
             if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    if len(data) == 0:
        return votation_entries(l_search)
    else:
        return votation_entries(data)

def return_highest_search(l_search):
    return votation_entries(l_search)

"""
