# 13 — arXiv sweep, Aug 2026

Searches: Arabic diacritization SOTA, GRPO variants, homograph G2P,
Hebrew vocalization. Papers read in full: GTPO/GRPO-S, HomoRich,
DIVRIT, Mohamed&Mubarak EMNLP 2025 (abstract+conclusions).

## Applied immediately

### GTPO entropy-weighted advantages — arXiv 2508.04349 (v6 Feb 2026)
Dynamic Entropy Weighting: per-token reward share ∝ token entropy, not
flat sequence reward. Diagnoses BOTH of our RL observations:
- Persian GRPO degrading past step 200 (3.71% → 4.38% by step 700):
  sequence reward smears credit over ~600 mostly-copied letters; the
  decisions are the few high-entropy homograph/haraqat positions. KL
  leash holds the policy near reference, but the noisy credit signal
  pushes it to garbage before the leash can matter.
- Arabic RAFT flat (same family of problem, positive-only variant).
WIRED IN: `train_arabic_grpo.py` token_logprobs now returns
`(lp * w).sum(1), w.sum(1)` with w = clamp(ent/mean_ent, 0.1, 4.0) * mask
(commit bfa6f82, branch sota-sprint-arabic). Persian v2 GRPO with the
same weighting is the queued follow-up if Arabic moves.

### Multi-reference evaluation — Mohamed & Mubarak, EMNLP 2025 (2025.emnlp-main.846)
WikiNews-2024: score against MULTIPLE valid gold readings instead of
one. Directly validates our diagnosis (98.7% of residual errors are
phonotactically legal alternates) — single-ref DER under-credits
everyone. Their data is already on our volume (QCRI EMNLP 2025 pull).
QUEUE: score r3/r5/GRPO-best under WikiNews-2024 multi-ref protocol.
This is an eval-protocol upgrade, zero training cost, potential big
reported-number movement, and a methods contribution for the paper
(their protocol + our windowed zero-skip projection can compose).
Also from that paper: "preserves user-specified diacritics" mode —
cheap product feature for the mini-model tier; low-resource data
augmentation gains — matches our replay/domain mix finding.

## Benchmark-line checks (claims safe)

### HomoRich — arXiv 2505.12973 (Fetrat Qharabagh, Dehghanian, Rabiee)
The group behind Homo-GE2PE (76.89%) and SentenceBench — the exact line
we benchmark against. Our Persian v1 77.34% ezafe-norm stays ahead of
their published best. Their new contributions: HomoRich dataset +
HomoFast eSpeak (rule-based, latency-sensitive tier). No new
DL-based number above 76.89 found → our lead claim holds as of this
sweep. Cite HomoFast as the "fast tier" prior art for our mini-model
distribution story.
Related: Bridging the Gap / intermediate language (2505.06599),
ParsHomo (T5 homograph disambiguation, 2025) — watch, not yet ahead.

### DIVRIT — arXiv 2510.26521 (Elboher & Pinter, Oct 2025)
Hebrew diacritization as zero-shot classification over a dynamically
generated candidate set, word-level, Hebrew Visual LM (text as image).
Oracle-candidate accuracy high → confirms the decomposition we already
run (trie/lexicon candidates + contextual discrimination). Not a
threat on full-text DER (word-level, oracle framing); cite in
paper-hebrew related work as concurrent candidate-classification work.

### Phonikud — arXiv 2506.12311
Unvocalized Hebrew → full IPA via nikud intermediate = exactly our
diacritizer→transliteration product chain. Cite in umbrella paper.
MenakBERT (2410.02417), D-Nikud (2402.00075): related work only.

## Watch list (no action yet)
- KSAA-2026 Fine-Tashkeel shared task — new baselines incoming.
- Consensus GRPO (2602.03102), F-GRPO (2605.12995): if GTPO alone
  doesn't move Arabic, factorized/two-phase GRPO is the next variant.
- PTCAD (2401.04848): token-classification framing; our char-encoder
  family already covers this.

## Queue after r5
1. r5 verdict (windowed zero-skip) → paper/RESULTS/MODELS
2. GTPO-GRPO (auto-fires) → if moves, Persian GRPO-v2 with weighting
3. WikiNews-2024 multi-ref eval (protocol upgrade, cheap)
4. r6 morph aux-task (qalsadi labels in flight)
5. Cross-abjad multi-task (note 03)
