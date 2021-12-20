#!/usr/bin/env python
# coding: utf-8


LIB_LOCAL_PATH = "/home/jair/WORK/Farsi/rababa"

import sys
import pandas as pd


sys.path.append(LIB_LOCAL_PATH)


import lib.get_assets as assets
import lib.model0_9 as l0_9
import lib.rules_and_coord as m0_9


df_test = pd.read_csv("data/test.csv")

df_test["trans0_9"] = [m0_9.run_transcription(d) for d in df_test["orig"]]


def evaluation(trans_orig, trans_model, orig):
    l_bugs = []
    tp, fp = 0, 0
    for i, d in enumerate(zip(trans_orig, trans_model)):
        l_orig = d[0].split()
        l_model = d[1].split()

        correct = True
        for o in zip(l_orig, l_model):
            if o[0] == o[1]:
                tp += 1
            else:
                fp += 1
                correct = False
        if not correct:
            l_bugs.append({"id": i, "trans": d[0], "trans_model": d[1]})

    print({"accuracy": tp / (tp + fp)})
    return [d["id"] for d in l_bugs]


ids = evaluation(df_test["trans"], df_test["trans0_9"], df_test["orig"])
df_bugs = df_test.iloc[ids]


print('error summary:')
print(df_bugs)
df_bugs.to_csv('tests/test_debug.csv')
