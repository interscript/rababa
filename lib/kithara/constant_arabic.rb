
"""
Constants that are useful for processing arabic diacritics.
reference:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py
"""

module Arabic_constant

    HARAQAT = ['\u0652', '\u0651', '\u064c', '\u064d', '\u0650', '\u064b', '\u064e', '\u064f']
    PUNCTUATIONS = ['.', '\u060c', ':', '\u061b', '-', '\u061f']
    ARAB_CHARS = 'ىعظحرسيشضق ثلصطكآماإهزءأفؤغجئدةخوبذتن'
    ARAB_CHARS_NO_SPACE = 'ىعظحرسيشضقثلصطكآماإهزءأفؤغجئدةخوبذتن'
    ARAB_CHARS_PUNCTUATIONS = ARAB_CHARS + PUNCTUATIONS.join('')
    VALID_ARABIC = HARAQAT + ARAB_CHARS.chars()

    BASIC_HARAQAT = {
        '\u064e'=> 'Fatha              ', # َ
        '\u064b'=> 'Fathatah           ', # ً
        '\u064f'=> 'Damma              ', # ُ
        '\u064c'=> 'Dammatan           ', # ٌ
        '\u0650'=> 'Kasra              ', # ِ
        '\u064d'=> 'Kasratan           ', # ٍ
        '\u0652'=> 'Sukun              ', # ْ
        '\u0651'=> 'Shaddah            ', # ّ
    }

    ALL_POSSIBLE_HARAQAT = {''=> 'No Diacritic       ',
                            '\u064e'=> 'Fatha              ',
                            '\u064b'=> 'Fathatah           ',
                            '\u064f'=> 'Damma              ',
                            '\u064c'=> 'Dammatan           ',
                            '\u0650'=> 'Kasra              ',
                            '\u064d'=> 'Kasratan           ',
                            '\u0652'=> 'Sukun              ',
                            '\u0651'=> 'Shaddah            ',
                            '\u0651\u064e'=> 'Shaddah + Fatha    ',
                            '\u0651\u064b'=> 'Shaddah + Fathatah ',
                            '\u0651\u064f'=> 'Shaddah + Damma    ',
                            '\u0651\u064c'=> 'Shaddah + Dammatan ',
                            '\u0651\u0650'=> 'Shaddah + Kasra    ',
                            '\u0651\u064d'=> 'Shaddah + Kasratan '}

end
