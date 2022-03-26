

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

    if type(l_search)==str:
        return "yes"

    if not pos is None:
        l_search_tmp = [d for d in l_search
                        if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    else:
        l_search_tmp = l_search

    return "yes" if len(set([d['PhonologicalForm'] for d in l_search_tmp])) == 1 else "no"

def votation_entries(l_search, entries=True):

    d_results = {}
    for item in l_search:
        form = item['PhonologicalForm']
        if d_results.get(form, False):
            d_results[form] += item.get('Freq', 1)
        else:
            d_results[form] = item.get('Freq', 1)

    M = max(d_results.values())
    idx = list(d_results.values()).index(M)
    return list(d_results.items())[idx][0]


def return_highest_search_pos(l_search, pos):

    if type(l_search) == str:
        return l_search

    data = [d for d in l_search
             if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    if len(data) == 0:
        return votation_entries(l_search)
    else:
        return votation_entries(data)

def return_highest_search(l_search):
    return votation_entries(l_search)



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


#def recu_entries(wrd, pos_pos=None, pos_neg=None):
#    #
#    #    Recursive search in entries_DB:
#    #    decompose wrd into largest substrings found in DB.
#    #
#    for i in range(len(wrd), 0, -1):
#        if df_Entries[df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
#            l_search = #df_Entries[df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
#           l_search = filter_search(l_search, pos_pos, pos_neg)
#            return votation_entries(l_search) + recu_entries(wrd[i:], #pos_pos=pos_pos, pos_neg=pos_neg)
#            break
#
#    return wrd


def recu_entries(wrd, pos_pos=None, pos_neg=None):
    #
    #    Recursive search in entries_DB:
    #    decompose wrd into largest substrings found in DB.
    #
    for i in range(len(wrd), 0, -1):
        if df_Entries[df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
            l_search = df_Entries[df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
 
            return votation_entries(l_search) + recu_affixes(wrd[i:])

    return wrd


def recu_affixes(wrd, pos_pos=None, pos_neg=None):
    # 
    #    Recursive search in affixes_DB:
    #    decompose wrd into largest substrings found in DB.
    #
    for i in range(len(wrd), 0, -1):
        if df_Affixes[df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            l_search = df_Affixes[df_Affixes['Affix']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            return votation_entries(l_search, entries=False) + recu_affixes(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
            break

    return wrd

"""
