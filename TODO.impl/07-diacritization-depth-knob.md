# 07 — Diacritization-depth conditioning (P3, product decision)

Source: TODO.qwen-next/10 §6. DeepSeek trains a scalar reasoning
effort (1–100) as an input-side signal; one checkpoint then serves
low/high/max tiers by conditioning. The analogous product surface for
us: one Arabic model with a depth control — none / light / full
tashkeel — instead of users post-filtering full output (which is what
"light" consumers do today).

## Why gated

- Needs a new student run with conditioning baked in (input-side
  marker or reserved control bytes) and a training corpus labeled at
  both depths; classical sources are fully voweled, so the "light"
  labels must be derived (strip a haraqat subset), which is a
  labeling decision with quality consequences.
- Product question first: do users ask for light tashkel from us, or
  is full output + client-side filtering enough?

## If built

- Control as data: a reserved input token (never a code path).
- Training: same recipe as the next student run, plus a depth field
  in the batch; eval protocol gains per-depth DER (light rows scored
  after the same projection).
- Runtime: an optional second argument, defaulted to full — OCP at
  the API boundary.

## Acceptance (only after the owner approves the product surface)

- [ ] Depth-label derivation rule documented and spot-checked
- [ ] One conditioned run, full-set per-depth verdicts
- [ ] API surface spec across TS/Python/Ruby

## Status

- [ ] Owner decision requested (registered from the memo, not built)
