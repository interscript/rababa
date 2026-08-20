# 03 — RAG homograph disambiguation probe (Persian first)

## Why
Persian v1 sits at 77.34% SentenceBench homograph (ezafe-normalized) —
above published Homo-GE2PE (76.89) but flat: RL was negative, the
mapped-representation line is closed. Retrieval context is the one
untried lever, and homograph resolution is precisely a context problem.

## Plan
1. `eval_persian_rag_probe.py` (rababa-farsi) — INFERENCE ONLY, no
   retraining for the probe:
   - Index HomoRich TRAIN homograph sentences (TF-IDF char n-gram
     retrieval over sentence contexts, pure numpy/sklearn-free).
   - For each SentenceBench test sentence, retrieve top-k (k=3)
     contextually nearest TRAIN sentences containing the same
     homograph token, format as few-shot prefix:
     `<train-sentence> => <diacritized/phonemized> ;` repeated, then
     the test sentence. ByT5-small handles prefix context in bytes.
   - Baseline vs RAG on the SAME harness (`eval_sentencebench.py`
     protocol, ezafe-normalized homograph accuracy + exact match).
2. Decision rule:
   - RAG ≥ +1.5pp homograph → invest: cache retrieval index, then a
     fine-tune WITH retrieved prefixes (train-time consistency).
   - +0.5..1.5pp → cheap inference-time add-on only, document.
   - < +0.5pp → close the lever, record negative in docs/RESULTS.md.
3. If Persian moves, port the probe to Hebrew (Nakdimon homographs)
   and Arabic (SadeedDiac residual analysis).

## Guards
- Retrieval from TRAIN splits only — zero test contamination; assert
  no test sentence appears in the index.
- Teacher stays v1 (RELEASE-FROZEN); probe never modifies it.
