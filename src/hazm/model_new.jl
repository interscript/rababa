

include("get_assets.jl")


py"""

def filter_search(l_search, pos_pos=None, pos_neg=None):

    if not pos_pos is None: # filter pos_pos only
        l_search_tmp = [d for d in l_search
                        if d_map_FLEXI.get(d['SynCatCode'], False)==pos_pos]
        if len(l_search_tmp) > 0:
            l_search = l_search_tmp

    if not pos_neg is None: # filter pos_neg out
        l_search = [d for d in l_search
                    if d_map_FLEXI.get(d['SynCatCode'], False)!=pos_neg]

    return l_search

def search_db(wrd, pos_pos=None):

    # all matches
    l_search = df_Entries[df_Entries['WrittenForm']==wrd].to_dict('records')
    l_search = filter_search(l_search, pos_pos)

    if len(l_search) == 0: # if not found, returns wrd
        return wrd
    else: # if one unique found, return the form
        return l_search

"""
