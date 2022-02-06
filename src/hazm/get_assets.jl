

using PyCall


py"""

from __future__ import unicode_literals
from hazm import *

import pandas as pd
import numpy as np
import _pickle as cPickle


LIB_LOCAL_PATH = '/home/jair/WORK/Farsi/rababa'

import sys

sys.path.append(LIB_LOCAL_PATH)


PATH_HAZM = "resources/postagger.model"

stemmer = Stemmer()
lemmatizer = Lemmatizer()
normalizer = Normalizer()
tagger = POSTagger(model=PATH_HAZM)


file = open('resources/farsi_assets.pickle','rb')
dic_assets = cPickle.load(file)
file.close()

df_Affixes = dic_assets['affixes']
df_Entries = dic_assets['entries']
d_FLEXI = dic_assets['d_FLEXI']
d_map_FLEXI = dic_assets['d_map_FLEXI']
l_PoS = list(set([s for s in d_map_FLEXI.values()]))
d_HAZM = dic_assets['d_HAZM']
d_map_HAZM = dic_assets['d_map_HAZM']

"""
