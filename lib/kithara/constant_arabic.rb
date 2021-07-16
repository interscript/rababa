
"""
Constants that are useful for processing arabic diacritics.
reference:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py
"""

module Arabic_constant

    HARAQAT = ['ْ', 'ّ', 'ٌ', 'ٍ', 'ِ', 'ً', 'َ', 'ُ']
    UHARAQAT = ['\u0652', '\u0651', '\u064c', '\u064d', '\u0650', '\u064b', '\u064e', '\u064f']
    PUNCTUATIONS = ['.', '\u060c', ':', '\u061b', '-', '\u061f']
    ARAB_CHARS = '\u0649\u0639\u0638\u062D\u0631\u0633\u064A\u0634\u0636\u0642\20\u062B\u0644\u0635\u0637\u0643\u0622\u0645\u0627\u0625\u0647\u0632\u0621\u0623\u0641\u0624\u063A\u062C\u0626\u062F\u0629\u062E\u0648\u0628\u0630\u062A\u0646'
    ARAB_CHARS_NO_SPACE = '\u0649\u0639\u0638\u062D\u0631\u0633\u064A\u0634\u0636\u0642\u062B\u0644\u0635\u0637\u0643\u0622\u0645\u0627\u0625\u0647\u0632\u0621\u0623\u0641\u0624\u063A\u062C\u0626\u062F\u0629\u062E\u0648\u0628\u0630\u062A\u0646'
    ARAB_CHARS_PUNCTUATIONS = ARAB_CHARS + PUNCTUATIONS.join('')
    VALID_ARABIC = HARAQAT + ARAB_CHARS.chars()

    BASIC_HARAQAT = {
        'َ': 'Fatha              ',
        'ً': 'Fathatah           ',
        'ُ': 'Damma              ',
        'ٌ': 'Dammatan           ',
        'ِ': 'Kasra              ',
        'ٍ': 'Kasratan           ',
        'ْ': 'Sukun              ',
        'ّ': 'Shaddah            ',
    }

    UBASIC_HARAQAT = {
        '\u064e'=> 'Fatha              ', # َ
        '\u064b'=> 'Fathatah           ', # ً
        '\u064f'=> 'Damma              ', # ُ
        '\u064c'=> 'Dammatan           ', # ٌ
        '\u0650'=> 'Kasra              ', # ِ
        '\u064d'=> 'Kasratan           ', # ٍ
        '\u0652'=> 'Sukun              ', # ْ
        '\u0651'=> 'Shaddah            ', # ّ
    }

    ALL_POSSIBLE_HARAQAT = {'': 'No Diacritic       ',
                            'َ': 'Fatha              ',
                            'ً': 'Fathatah           ',
                            'ُ': 'Damma              ',
                            'ٌ': 'Dammatan           ',
                            'ِ': 'Kasra              ',
                            'ٍ': 'Kasratan           ',
                            'ْ': 'Sukun              ',
                            'ّ': 'Shaddah            ',
                            'َّ': 'Shaddah + Fatha    ',
                            'ًّ': 'Shaddah + Fathatah ',
                            'ُّ': 'Shaddah + Damma    ',
                            'ٌّ': 'Shaddah + Dammatan ',
                            'ِّ': 'Shaddah + Kasra    ',
                            'ٍّ': 'Shaddah + Kasratan '}

    UALL_POSSIBLE_HARAQAT = {''=> 'No Diacritic       ',
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
