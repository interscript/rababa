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


#def logic():
