"""
Constants that are used by the model
"""

HARAQAT = ["ْ", "ّ", "ٌ", "ٍ", "ِ", "ً", "َ", "ُ"]
ARAB_CHARS = "\u0649\u0639\u0638\u062d\u0631\u0633\u064a\u0634\u0636\u0642 \u062b\u0644\u0635\u0637\u0643\u0622\u0645\u0627\u0625\u0647\u0632\u0621\u0623\u0641\u0624\u063a\u062c\u0626\u062f\u0629\u062e\u0648\u0628\u0630\u062a\u0646"
PUNCTUATIONS = [".", "،", ":", "؛", "-", "؟"]
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
