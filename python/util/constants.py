"""
Constants that are used by the model
"""
HARAQAT = ["ْ", "ّ", "ٌ", "ٍ", "ِ", "ً", "َ", "ُ"]
ARAB_CHARS = '\u0649\u0639\u0638\u062D\u0631\u0633\u064A\u0634\u0636\u0642 \u062B\u0644\u0635\u0637\u0643\u0622\u0645\u0627\u0625\u0647\u0632\u0621\u0623\u0641\u0624\u063A\u062C\u0626\u062F\u0629\u062E\u0648\u0628\u0630\u062A\u0646'
# PUNCTUATIONS = [".", "،", ":", "؛", "-", "؟"]
PUNCTUATIONS = list(set([".", "،", ":", "؛", "-", "؟"] +
                        [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?']))

VALID_ARABIC = HARAQAT + list(ARAB_CHARS) + [".", "،", ":", "؛", "-", "؟"]
BASIC_HARAQAT = {
    "َ": "Fatha  ",
    "ً": "Fathatah           ",
    "ُ": "Damma              ",
    "ٌ": "Dammatan           ",
    "ِ": "Kasra              ",
    "ٍ": "Kasratan           ",
    "ْ": "Sukun              ",
    "ّ": "Shaddah            ",
}
ALL_POSSIBLE_HARAQAT = {
    "": "No Diacritic       ",
    "َ": "Fatha              ",
    "ً": "Fathatah           ",
    "ُ": "Damma              ",
    "ٌ": "Dammatan           ",
    "ِ": "Kasra              ",
    "ٍ": "Kasratan           ",
    "ْ": "Sukun              ",
    "ّ": "Shaddah            ",
    "َّ": "Shaddah + Fatha    ",
    "ًّ": "Shaddah + Fathatah ",
    "ُّ": "Shaddah + Damma    ",
    "ٌّ": "Shaddah + Dammatan ",
    "ِّ": "Shaddah + Kasra    ",
    "ٍّ": "Shaddah + Kasratan ",
}

def decision_fct(d_haraqats):
    """
        {'fatha': char,
         'shaddah': char,
         'haraqat': char}
    """
    if d_haraqats['fatha'] != '':
        if d_haraqats['shaddah'] != '':
            return d_haraqats['fatha'] + d_haraqats['shaddah']
        else:
            return d_haraqats['fatha']
    if d_haraqats['haraqat'] == "ْ": # if sukun return sukun
        return "ْ"
    if d_haraqats['shaddah'] != '':
        return d_haraqats['shaddah'] + d_haraqats['haraqat']
    else:
        return d_haraqats['haraqat']
    return ''

"""
['haraqat', 'shaddah', 'fatha']
d_distr = {#'': 1.011542076671533,
    'َ': 0.17651175400202376,
    'ً': 0.003504662636992608,
    'ُ': 0.04588558786185574,
    'ٌ': 0.0032528178335757894,
    'ِ': 0.07471201949874581,
    'ٍ': 0.004594122977269162,
    'ْ': 0.05232150773113324,
    'ّ': 0.025341241600217505,
    'َّ': 0.017926102702184244,
    'ًّ': 0.00018712229821031827,
    'ُّ': 0.002690966916374738,
    'ٌّ': 0.00021909260950695207,
    'ِّ': 0.003615545174995154,
    'ٍّ': 0.0002617403759610149
}

l_distr = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ', 'َّ', 'ًّ', 'ُّ', 'ٌّ', 'ِّ', 'ٍّ']
"""
