
"""
Constants that are useful for processing arabic diacritics.
reference:
https://github.com/almodhfer/diacritization_evaluation/blob/master/diacritization_evaluation/constants.py
"""

module arabic_constant

    HARAQAT = ['ْ', 'ّ', 'ٌ', 'ٍ', 'ِ', 'ً', 'َ', 'ُ']
    PUNCTUATIONS = ['.', '،', ':', '؛', '-', '؟']
    ARAB_CHARS = 'ىعظحرسيشضق ثلصطكآماإهزءأفؤغجئدةخوبذتن'
    ARAB_CHARS_NO_SPACE = 'ىعظحرسيشضقثلصطكآماإهزءأفؤغجئدةخوبذتن'
    ARAB_CHARS_PUNCTUATIONS = ARAB_CHARS + ''.join(PUNCTUATIONS)
    VALID_ARABIC = HARAQAT + list(ARAB_CHARS)

    BASIC_HARAQAT = {
        'َ'=> 'Fatha              ',
        'ً'=> 'Fathatah           ',
        'ُ'=> 'Damma              ',
        'ٌ'=> 'Dammatan           ',
        'ِ'=> 'Kasra              ',
        'ٍ'=> 'Kasratan           ',
        'ْ'=> 'Sukun              ',
        'ّ'=> 'Shaddah            ',
    }

    ALL_POSSIBLE_HARAQAT = {''=> 'No Diacritic       ',
                            'َ'=> 'Fatha              ',
                            'ً'=> 'Fathatah           ',
                            'ُ'=> 'Damma              ',
                            'ٌ'=> 'Dammatan           ',
                            'ِ'=> 'Kasra              ',
                            'ٍ'=> 'Kasratan           ',
                            'ْ'=> 'Sukun              ',
                            'ّ'=> 'Shaddah            ',
                            'َّ'=> 'Shaddah + Fatha    ',
                            'ًّ'=> 'Shaddah + Fathatah ',
                            'ُّ'=> 'Shaddah + Damma    ',
                            'ٌّ'=> 'Shaddah + Dammatan ',
                            'ِّ'=> 'Shaddah + Kasra    ',
                            'ٍّ'=> 'Shaddah + Kasratan '}

end
