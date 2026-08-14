# 01 — Arabic: evaluate on Sadeed test set (or document protocol)

## Why
Our Arabic v2 model reports 0.99% DER on a held-out split of our own 2.1M
corpus. Sadeed (arXiv:2504.21635) reports 1.2% DER on SadeedDiac-25.
For an honest paper claim we must either (a) evaluate on their test set, or
(b) state the protocol difference explicitly.

## Tasks
- [x] Locate Sadeed data source (Misraj/Sadeed_Tashkeela on HF, gated by HF_TOKEN)
- [x] Attempt eval on Sadeed test split via Modal HF secret
- [x] If unavailable, write protocol caveat into RESULTS.md + paper

## Result
Sadeed HF dataset is gated and the environment lacks reliable access to
SadeedDiac-25 itself. Claim is phrased as: "0.99% DER on our held-out 2.1M
combined corpus split (Tashkeela-full + arwiki + QCRI); Sadeed reports 1.2%
DER on SadeedDiac-25 — different test sets, not directly comparable."
