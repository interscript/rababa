
"""
Constants that are useful for calculating DER and WER
"""

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
