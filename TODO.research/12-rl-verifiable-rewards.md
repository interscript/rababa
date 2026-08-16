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
