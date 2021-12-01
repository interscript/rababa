

# Farsi Rules & Implementation

### Rules

1. If a word can be found in the database, there's no need to break it down (stem for nouns and lemmatize for verbs)

2. For nouns that aren't found in the database, we need to use the stemmer function to get roots of the nouns. Then, much like with verbs, we should look and see which affixes were omitted from the word, find them in the Affixes table, and put them in proper positions.

3. When breaking down verbs to their roots using lemmatizer function, we get two outputs. ONLY ONE of them exists in the verb we have broken down. So, we only use that one. Then, we look to see which affixes were omitted from the verb when we broke it down, and we search for them in the Affixes table and add their transliteration to the root of the verb in proper positions, i.e. prefixes get added to the beginning of the word and suffixes, to its end.

4. For collisions, if the PoS tagging doesn't help, please use the transliteration with the higher frequency.

5. For verbs that contain underscores after being lemmatized, we need to look for the parts separately. So, for خواهند كرد, you get the output خواهند_كرد from the lemmatizer. Now, if we look up خواهند and كرد in the dataset separately, we find the latter one and use it directly (it's a collision, but since we know it's a verb, we use the proper one which is /kard/), and for خواهند, we can lemmatize it again. The output will be خواست#خواه. Since we have خواه in the verb, we look it up in the dataset and use the transliteration /xAh/. Then we see that ند has been omitted from the verb, so we look it up in the Affixes table, and add that (/and/) to the end of the verb root so the whole word will be /xAhand/ which is the concatenation of the two in proper order. So, in the end, the transliteration of خواهند كرد will be /xAhand kard/ Please note that the space is not added. It existed in the middle of خواهند كرد, and the underscore just reminded us of the fact that they are separate words.

6. The library works only with standard Farsi letters {> stemmer.stem('گزارشی')'گزارش' > stemmer.stem('گزارشي') 'گزارشي')} and I already converted entries, affixes, etc. to the standard letters. So, from now on, replace the letters ي and ك in inputs to ی and ک.

7. The affix ی, always a suffix, never a prefix, is used with verbs and nouns. For verbs, it has only one transliteration. If the word before it, is not a verb, which will usually be a noun (not always), if the word before it ends in ا or و or ه, the last transliteration of the noun kind, /ye/, should be used, else, the first transliteration should be used. I added PoS tags to the last two ones since they're for verbs.

8. Wherever there's the letter ۀ as in the word همۀ the transliteration of the word containing it should end in /eye/ based on the database transliteration system. That letter can also be written down as هٔ as in the word همهٔ too. So it may consist of one or two characters. Note that the word containing this letter already ends in /e/ so we only need to add the /ye/ part to the end of it.

9. The affix ات is a collision. Note that it's always a suffix, never a prefix. When faced, if it's attached to the word with no space or semispace, the first transliteration should be chosen. If there's a space or semispace before it, then choose the second transliteration.

10. The affix ان is a collision. Two of its instances are marked for nouns. It's always a suffix, never a prefix and to choose between the two, when it's attached to a noun, if the noun ends in ی, choose the last transliteration, else, choose the first transliteration.

11. The affix ش, always a suffix, is also a collision. If it's attached to a noun, choose the first transliteration, if it's anything else, choose the second transliteration.

12. The affix م, always a suffix, is a collision. According to the PoS tagging in affixes, the different transliteration should be used with numbers, and if the word before it isn't a number, the main transliteration should be used, UNLESS the word before it is چند. For this word, the different transliteration should be used.

13. The affix مان is a collision. When it's a suffix, attached to a noun of any type, (I've already sent a PoS mapping between our database and the library) choose the first transliteration.

14. The affix می is a collision. If it's a prefix, we know it's part of a verb and we use the second transliteration. If it's a suffix, we use the first transliteration even if the word before it isn't a number.

15. If we lemmatize a verb, but none of the two results are part of the verb we have lemmatized, it's possible the root of the verb has an آ letter that's been mentioned in our verb in ا form. Make sure to check that if such a case happens.

16. The affix ون, always a suffix, is a collision. To choose between the two, if the word before it, ends in ی, choose the second transliteration, else, choose the first transliteration.

17. I added a row to the affixes: نمی. I believe this will help. Will reconsider if it turns out to cause bugs somehow.

18. The affix ید, always a suffix, is a collision. If it's not attached to a verb, (so it doesn't need to be a noun. It can be an adjective, etc.) there's only one transliteration (the first one) But if it's attached to a verb, if the root of the verb before it has been the first output of the lemmatize function, always use the first transliteration, /id/, but if it has been the second output of the lemmatize function, {if it ends in ا or و with u sound, or ه, the second transliteration should be used, /yad/, else, the first transliteration, /id/, should be used} I added PoS tags to this affix in the proper column.

19. The affix یم, always a suffix, is a collision. PoS tagging really helps with this. I added PoS tags to the proper column for different instances of it. Basically, if it's used as part of a verb, the second transliteration should be used. Else, even if the PoS is not noun, the first transliteration should be used.

20. Semispace (ZWNJ): Wherever we see the character u200c, we should break the sides of it and work on them separately. The point, though, is that most of the time, it breaks affixes from roots, but not always. So, when we encounter one, we would wanna check for the parts separately. Most of the time when we see this character, there's going to be می or نمی before it, or ها after it. One point to consider here, is that می and نمی, which are verb prefixes, are exactly these values, but ها which is a noun suffix, is the beginning of a string, so it may not be exactly ها. but it starts with ها. If these aren't found next to the u200c charcater, we should look for both sides of it in entries.

21. If after breaking down a word into its root, the remaining parts aren't found in affixes, we should look for matching substrings. For example, the noun کتاب‌هایتان is stemmed into کتاب which can be found is entries. The remaining part, هایتان, however, is not among affixes. So, what we should do in these cases, is to look for the longest substring of the remaining part that can be found in affixes starting where the string starts. So, while we can go with the affix ه which is the starting letter of the remaining string, we won't, because ها which is a longer substring, can be found in affixes, too. Now why won't we go with یتان that is a substring of the remaining string and is actually even longer? Because it doesn't start where the remaining string starts. So, we go with ها and find it in affixes and add the transliteration to the root of the word. Then, we do the same with the remaining string, i.e. the string that is not transliterated yet. So, this is a recursive algorithm. In the case of this example, we find یتان in the affixes. Done! Note that this goes for verbs and lemmatization, too. So, we don't stop until the whole word is transliterated.

22. I added another row for ن which is pronounced /an/ because it didn't exist in the affixes while it's necessary for transliterating words like خوردن and خوابیدن that are called مصدر in Farsi.

23. The affix ن is a collision. It's always part of a verb. When it's a prefix, use the first transliteration and when it's a suffix, use the last transliteration, the one I added.

###  Implementation

1.  ok
2. ok
3. ok
4. ok
6.
7.
8.
9.
10.
11.
12.
13.
14.
15.
16.
17.
18.
19.
20. ok
21.
22.
23.
