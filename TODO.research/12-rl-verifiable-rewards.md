# 12 — RL fine-tuning with verifiable rewards (GLM-5.3 playbook)

Source: z.ai/blog/glm-5.3 — "scaling post-training" on a frozen base,
with binary rewards from verified environments.

## Why for us
Our metrics (DER/CER/HA vs references) are perfect verifiable rewards —
the exact regime RL beats SFT in. SFT has plateaued exactly where RL
should help: Arabic case endings (3.25 vs Claude 1.39) and Persian
homograph accuracy (77.34 vs recipe roulette).

## Steps
1. GRPO-style RL after SFT on ByT5 Arabic r2: reward = -DER per
   paragraph against corpus refs (oracle); no-op check = reward for
   copying input must be << true diacritization; unsolved check =
   shuffled targets score 0.
2. Same for Persian v1 with reward = homograph-exact (word-position).
3. Guard against reward shortcuts with their oracle/no-op/unsolved
   triple before trusting any reward.
4. TRL/verl or a hand-rolled GRPO loop; A100, watchdog+resume harness.

## Plus (same blog)
- Private per-language dev sets — stop selecting recipes on the public
  benchmark (v1>v4 flip is a selection-on-test signature).
- On-policy distillation of ensembles (OPD), NOT output voting.
- Training-rollout consistency checklist pinned in the eval harness
  (tokenizer/prefix/normalization) — industrial version of bugs we hit.

## Success metric
Arabic DER(CE) < 1.5 without touching the benchmark during development;
Persian SB HA >= 80 measured once, at the end.


## Final outcome (2026-08-19): all three RL variants negative

| Method | Language | Result |
|---|---|---|
| RAFT (positive-only) | Arabic | flat (2.8429 -> 2.8515 beam-4 / 2.8308 beam-1) |
| RAFT (positive-only) | Persian | tie (76.85 vs 77.34) |
| GRPO (sequence reward) | Persian | negative (75.37 vs 77.34; degraded past step 200) |
| GTPO-GRPO (entropy-weighted, graded reward) | Arabic | exactly flat dev (5.9692 x3); benchmark = r5 within protocol noise |

Verdict: knowledge-limited, not policy-limited. Posterior sharpening
cannot recover distinctions the SFT posterior has already baked in from
2M clean lines. Infrastructure lessons recorded in git history: binary
reward was dead at temp 1.0 (all-1.0 groups), byt5-base GRPO needs
A100-80GB or gradient checkpointing at 700B+ units, and flat curves
must still materialize best/ before the final eval.
