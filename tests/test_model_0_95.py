#!/usr/bin/env python
# coding: utf-8

LIB_LOCAL_PATH = '/home/jair/WORK/Farsi/rababa'


import sys
sys.path.append(LIB_LOCAL_PATH )


import lib.get_assets as assets
import lib.model0_9 as l0_9
import lib.rules_and_coord as m0_9


# ### Rule 1

# In[48]:


tests = [
    {"farsi": "ییلاق‌نشین", "trans": "yeylAqneSin"},
    {"farsi": "ییدیش", "trans": "yidiS"},
    {"farsi": "یوونتوس", "trans": "yuventus"},
    {"farsi": "یونید", "trans": "یونید"},
]  # case verb

for t in tests:
    assert l0_9.general_search(t["farsi"], pos_neg="Verb") == t["trans"]


# In[ ]:


# ### Rule 2 (Nouns)

# In[49]:


tests = [
    {"farsi": "پیامبرت", "trans": "payAmbarat"},
    {"farsi": "مزايایی", "trans": "mazAyAyi"},
    # {'farsi': 'بی شرف', 'trans': 'bi Saraf'},
    # {'farsi': 'بی عقل', 'trans': 'bi \'aql'}
]

for t in tests:
    # print(m0_9.run_transcription(t['farsi']))
    assert m0_9.run_transcription(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 3 (Verbs basis)

# In[50]:


tests = [
    {"farsi": "بخواهند بروند", "trans": "bexAhand beravand"},
    # {'farsi': 'می‌تواند بخوابد', 'trans': 'mitavAnad bexAbad'}, # \u200c
    # {'farsi': 'نشد بپریم', 'trans': 'naSod beparim'}, # می
    # {'farsi': 'می روید', 'trans': 'miravid'} #
    # {'farsi': 'می ترسد', 'trans': 'mitarsad'}
]

for t in tests:
    assert m0_9.run_transcription(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 4 (Frequency based prioritization for collisions)

# In[51]:


tests = [{"farsi": "ی", "trans": "ye"}]

for t in tests:
    assert l0_9.general_search(t["farsi"]) == t["trans"]


tests = [{"farsi": "ی", "trans": "i"}]
for t in tests:
    assert l0_9.affix_search(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 5 (Verb _ as خواهند_كر)

# In[52]:


tests = [{"farsi": "خواهند_كرد", "trans": "xAhand kard"}]

for t in tests:
    assert l0_9.process_verb(l0_9.normalise(t["farsi"])) == t["trans"]

tests = [
    {"farsi": "خواهند كرد", "trans": "xAhand kard"},
    {"farsi": "بخواهند بروند", "trans": "bexAhand beravand"},
]

for t in tests:
    assert m0_9.run_transcription(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 6 (normalisation ي, ك)

# In[53]:


tests = [{"farsi": "گزارشي", "trans": "گزارشی"}, {"farsi": "ك", "trans": "ک"}]

for t in tests:
    assert l0_9.normalise(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 7 (affix ی)

# In[54]:


tests = [  # {'farsi': 'بانوی', 'trans': 'bAnuye', 'pos': 'Noun', 'pos_last': 'Adjective'},
    {"farsi": "چیزی", "trans": "Cizi", "pos": "Noun", "pos_last": "Verb"}
]

for t in tests:
    assert m0_9.rule_7(t["farsi"], pos=t["pos"], pos_last=t["pos_last"]) == t["trans"]


# In[ ]:


# ### Rule 8 (letter ۀ)

# In[55]:


tests = [
    {"farsi": "همۀ", "trans": "hameye", "pos": "Determiner"},
    {"farsi": "بچهٔ", "trans": "baCCeye", "pos": "Verb"},
]

for t in tests:
    assert m0_9.rule_8(t["farsi"], pos=t["pos"]) == t["trans"]


# In[ ]:


# ### Rule 9 (collision affix ات)

# In[56]:


tests = [
    {"farsi": "مزخرفات", "trans": "mozaxrafAt"},
    {"farsi": "خاطرات‌مان", "trans": "xAterAtemAn"},
    {"farsi": "عمه‌ات", "trans": "'amme'at"},
]


for t in tests:
    assert m0_9.rule_9(l0_9.normalise(t["farsi"])) == t["trans"]


# In[ ]:


# ### Rule 10 (collision affix ان)

# In[57]:


tests = [
    {"farsi": "بانیان", "trans": "bAniyAn", "pos": "Noun"},
    {"farsi": "دیگران", "trans": "digarAn", "pos": "Preposition"},
]


for t in tests:
    assert m0_9.rule_10(l0_9.normalise(t["farsi"])) == t["trans"]


# In[ ]:


# ### Rule 11 (collision affix ش)

# In[58]:


tests = [
    {"farsi": "وسایلش", "trans": "vasAyelaS", "pos": "Noun"},
    {"farsi": "بردندش", "trans": "bordandeS", "pos": "Verb"},
]

for t in tests:
    assert m0_9.rule_11(t["farsi"], t["pos"]) == t["trans"]


# In[ ]:


# ### Rule 12 (affix م)

# In[59]:


tests = [{"farsi": "چندم", "trans": "Candom", "pos": "Number"}]

for t in tests:
    assert m0_9.process_wrd(t["farsi"], t["pos"]) == t["trans"]


# In[ ]:


# ### Rule 13 (affix مان)

# In[60]:


tests = [
    {"farsi": "برای‌مان", "trans": "barAyemAn", "pos": "Preposition"},
    {"farsi": "خاطرات‌مان", "trans": "xAterAtemAn", "pos": "Noun"},
    # {'farsi': 'کردن‌مان', 'trans': 'kardanemAn', 'pos': 'Noun'}
]

for t in tests:
    # print(rule_13(t['farsi'], t['pos']))
    assert m0_9.rule_13(t["farsi"], t["pos"]) == t["trans"]


# In[ ]:


# ### Rule 14 (affix می)

# In[61]:


tests = [  # {'farsi': 'می‌تواند', 'trans': 'mitavAnad', 'pos': 'Verb'},
    {"farsi": "چندمی", "trans": "Candomi"}  # ,
]

for t in tests:
    # print(m0_9.rule_14(t['farsi'], t.get('pos', None)))
    assert m0_9.rule_14(t["farsi"], t.get("pos", None)) == t["trans"]


# In[ ]:


# ### Rule 15 (affix آ)

# In[62]:


text = "بیا در آغوشم بیارام و دیگران را نیازار"
# 'biyA dar 'AquSam biyArAm va digarAn rA nayAzAr'
# assets.tagger.tag(assets.word_tokenize(text))


# In[ ]:


# ### Rule 16 (affix ون)

# In[63]:


tests = []  # {'farsi': 'سرنگون', 'trans': 'sarnegun', 'pos': 'Adjective'},
# {'farsi': 'حواریون', 'trans': 'havAriyun', 'pos': 'Verb'},
# {'farsi': 'منافقون', 'trans': 'monAfequn', 'pos': 'Noun'}]


for t in tests:
    # print(m0_9.rule_16(t['farsi'], t.get('pos', None)))
    assert m0_9.rule_16(t["farsi"], t.get("pos", None)) == t["trans"]


# In[ ]:


# ### Rule 17 (affix ید)

# In[64]:


tests = [
    {"farsi": "رفتید", "trans": "raftid", "pos": "Verb"},
    {"farsi": "بگوید", "trans": "beguyad", "pos": "Verb"},
    # {'farsi': 'می‌آید', 'trans': 'beguyad', 'pos': 'Verb'}
]

for t in tests:
    assert m0_9.rule_17(t["farsi"], t.get("pos", None)) == t["trans"]


# In[ ]:


# ### Rule 18 (verb with یم)

# In[65]:


tests = [
    {"farsi": "بپریم", "trans": "beparim", "pos": "Verb"},
    {"farsi": "رفتیم", "trans": "raftim", "pos": "Verb"},
    {"farsi": "خدایم", "trans": "xodAyam", "pos": "Noun"},
]

for t in tests:
    assert m0_9.process_wrd(t["farsi"], t.get("pos", None)) == t["trans"]


# In[ ]:


# ### Rule 19 (Semispace u200c, improve imlementation in 0.9)

# In[66]:


tests = [  # {'farsi': 'دیده_می\u200cشود', 'trans': 'dide miSavad', 'pos': 'Verb'},
    {"farsi": "بی\u200cبصیرتی\u200cهایی", "trans": "bibasiratihAyi", "pos": "Noun"},
    {"farsi": "چراغ‌علی", "trans": "CerAq'ali", "pos": "Noun"},
]

for t in tests:
    # print(rule_19(t['farsi'], t['pos']))
    assert m0_9.rule_19(t["farsi"], t["pos"]) == t["trans"]


# In[ ]:


# ### Rule 20 (Recursive search, improve implemented in 0.9)

# In[67]:


"""
def recu_affix_wrd(wrd):

    suffix, stem = '', ''
    for i in range(len(wrd), 0, -1):
        if assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            l_search = assets.df_Affixes[assets.df_Affixes['Affix']==wrd[:i]].to_dict('records')
            stem = m0_9.votation_entries(l_search, entries=False)
            suffix = recu_affix_wrd(wrd[i:])
            break

    return stem + suffix


tests = [{'farsi': 'هایی', 'trans': 'hAyi'},
         {'farsi': 'هایتان', 'trans': 'hAyetAn'}]

for t in tests:
    assert l0_9.recu_affix_wrd(t['farsi']) == t['trans']
"""


# In[ ]:


# ### Rule 21 (affix ن)

# In[68]:


tests = [
    {"farsi": "نخوردن", "trans": "naxordan"},
    {"farsi": "نخوابیدن", "trans": "naxAbidan"},
    # {'farsi': 'نمان', 'trans': 'namAn'}
]

for t in tests:
    # print(rule_21(t['farsi']))
    assert m0_9.rule_21(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 22 (affixes بی and نی)

# In[69]:


tests = [
    {"farsi": "بیا", "trans": "biyA"},
    # {'farsi': 'نیازار', 'trans': 'nayAzAr'},
    # {'farsi': 'نمان', 'trans': 'namAn'}
    # {'farsi': 'بیارام', 'trans': 'biyArAm'}
]

for t in tests:
    # print(rule_22(t['farsi']))
    assert m0_9.rule_22(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 23 (root of the verb is رو)

# In[70]:


tests = [{"farsi": "نرو", "trans": "naro"}, {"farsi": "برو", "trans": "bero"}]

for t in tests:
    assert m0_9.rule_23(t["farsi"]) == t["trans"]


# In[ ]:


# ### Rule 24 (Recursion, improve v0.9)

# In[71]:


tests = [{"farsi": "فیسبوک", "trans": "fisbuke"}]

for t in tests:
    assert l0_9.recu_entries(t["farsi"]) == t["trans"]

tests = [{"farsi": "ازار", "trans": "AzAr"}]

for t in tests:
    assert l0_9.recu_affixes(t["farsi"]) == t["trans"]


# In[ ]:
print('ran unit tests successfully')
