from util.constants import HARAQAT


"""
##################
# ALGORITHM IDEA #
##################

Given strings: s_original and s_diacritized 
1. Build pivot map of matches, so for instance:
    (24 abc, axbycz) -> [(3,0), (4,2), (5,4)]
2. Build a string from "pivots":
    a. first with the original string if needed
    b. then with diacritized
3. finalize by writing:
    a. end of diacritics
    b. end of original
"""

def build_pivot_map(d_original, d_diacritized):
    """build_pivot_map:
        This function takes 2 strings and finds the "pivot points", 
        i.e the points where both strings are identical. 
        args:
            d_original: dictionary modelling the original string abc -> {0:a,1:b,2:c}
            d_diacritized: dictionary modelling diacritized as above 
        return: list of ids tuple where strings match
    """ 
    l_map = []
    idx_dia, idx_ori = 0, 0
    while idx_dia < len(d_diacritized):
    
        c_dia = d_diacritized[idx_dia]
        for i in range(idx_ori, len(d_original)):
            if c_dia == d_original[i]:
                idx_ori = i
                l_map.append((idx_dia, idx_ori))
                break

        idx_dia += 1

    return l_map


def reconcile_strings(str_original, str_diacritized):
    """reconcile_strings:
        This function takes original and diacritized string and merge them into a sensible output.
        For instance: 
            original string: 
                # گيله پسمير الجديد 34
            diacritised string (with non arabic removed by the nnets preprocessing):
                يَلِهُ سُمِيْرٌ الجَدِيدُ 
            reconcile_strings -->
                '# گيَلِهُ پسُمِيْرٌ الجَدِيدُ 34'
        
        Other examples and tests can be found in the commented section below.
        args:
            str_original: original string
            str_diacritized: diacritized string 
        return: reconciled string
    """
    # we model the strings as dict
    d_original = dict((i,c) for i,c in 
                      enumerate(list([c for c in str_original if not c in HARAQAT])))
    d_diacritized = dict((i,c) for i,c in enumerate(list(str_diacritized)))

    # matching positions
    l_pivot_map = build_pivot_map(d_original, d_diacritized)
    
    str__ = '' # "accumulated" chars 
    pt_dia, pt_ori = 0, 0 # pointers for resp diacr and orig. strings
    for x_dia, x_ori in l_pivot_map:        
                
        # We start to write characters from original strings
        if pt_ori < x_ori:
            for i in range(pt_ori,  x_ori):
                str__ += d_original[i]

        # We then add chars from diacritized strings
        if pt_dia < x_dia:
            for i in range(pt_dia,  x_dia):
                str__ += d_diacritized[i]
        # append matches
        str__ += d_original[x_ori]
        pt_dia, pt_ori = x_dia + 1, x_ori + 1

    # Finalize by adding first last diacritized chars and then
    for i in range(pt_dia,  len(d_diacritized)):
        str__ += d_diacritized[i]
        
    # remaining chars for original string        
    for i in range(pt_ori,  len(d_original)):
        str__ += d_original[i]
    
    return str__


"""
###############
# TEST SCRIPT #
###############

import test

import util.reconcile_original_plus_diacritized as reconcile

d_tests = [{'original': '# گيله پسمير الجديد 34', 
            'diacritized': 'يَلِهُ سُمِيْرٌ الجَدِيدُ',
            'reconciled': '# گيَلِهُ پسُمِيْرٌ الجَدِيدُ 34' },
           
           {'original': 'abc',
            'diacritized': '',
            'reconciled': 'abc'},
           
           {'original': '‘Iz. Ibrāhīm as-Sa‘danī',
            'diacritized': '',
            'reconciled': '‘Iz. Ibrāhīm as-Sa‘danī'},
           
           {'original': '26 سبتمبر العقبة', 
            'diacritized': 'سَبْتَمْبَرِ العَقَبَة', 
            'reconciled': '26 سَبْتَمْبَرِ العَقَبَة'}]

for d in d_tests:
    str_reconciled = reconcile.reconcile_strings(d['original'], d['diacritized'])
    assert str_reconciled == d['reconciled'], 'pbm with strings reconciliation'

"""
