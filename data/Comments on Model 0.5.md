### 1. Left to right outputs seem to be easier to evaluate, like model 0.3 outputs, especially in the debugging files. Please go back to left to right.

### 2. One thing I found is that the word Ïæ is truly recognized as a number, but the wrong transliteration is chosen from the dataset.

### 3. For collisions, if the PoS tagging doesn't help, please use the transliteration with the higher frequency. For example, the word ÇÚáÇã is a collision and both pronunciations are nouns. So, PoS tagging doesn't help us figure out which one to use. For now, please use the one with the higher frequency, which is /'e'lAm/