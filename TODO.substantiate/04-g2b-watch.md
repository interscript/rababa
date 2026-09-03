# 04 — G2b verdict watch + cross-recording

Priority: P2. Status: OPEN (dependent on other agents' run).

G2b (48k units from full Tashkeela — the ADD direction) is in flight
on the auto-chain. E6's swap-negative makes G2b the decisive test of
the domain-coverage hypothesis: if ADD helps where SWAP hurt, the
residual was budget-limited domain coverage; if ADD also fails, the
residual is not domain-shaped at all.

## Steps

- [ ] Watch for final_eval.json under the G2b run dir (checkpoints
      volume, spec per ml EXPERIMENTS.md G2b registration)
- [ ] When it lands: verify against its registered gate; record the
      E6-swap vs G2b-add pair in PUBLICATION-NOTES Paper-B framing
- [ ] Do NOT duplicate-launch anything (other agents own the run;
      modal app list showed their ephemeral apps)
