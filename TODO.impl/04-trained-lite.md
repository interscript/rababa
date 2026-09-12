# 04 — lite-2.0: init the depth cut from the trained student (P1)

Source: TODO.qwen-next/10 §4. CORRECTED 2026-09-11 (spec audit):
the shipped lite rung (run-009-layerdrop-6ep, 5.78) IS a trained
6-epoch sequence-KD distill — its layer-drop INIT copies kept layers
verbatim from generic pretrained byt5-small. The untested variable in
the lite cell is the INIT SOURCE: DeepSeek's repeated pattern is
compress-from-TRAINED-weights then adapt (CSA2 modes are trained from
the start; Single-Pass mHC shifts a dependency and eats it in
training). Dropping layers from the trained 2.1 student instead of
generic byt5-small is the single-variable variant.

## Experiment

`ara-diac-small-lite2`: byte-identical to the run-009 spec except
`student_init: /checkpoints/rababa_arabic_distill_small/run-007-r7-muon-6ep/best`
(the shipped 2.1) — the layer_drop bridge then copies its kept layers.

- Gate: full-set DER vs 5.78 (run-009) with paired bootstrap; also
  compare against 2.1 (4.57) to locate the lite ceiling.
- The Hebrew collapse (77.48) is a separate cross-lingual replication
  failure with a known recipe confound (logit-KD 3ep vs Arabic
  sequence-KD 6ep) — not evidence about this axis.

## Why this ordering

Init source is the one lever in the lite cell with a mechanism story
(task-adapted features survive the drop better than generic ones) and
a cost of one run. If it fails, the lite frontier is recipe-bound and
/10 (lexical memory) becomes the remaining architecture lever.

## Acceptance

- [ ] Spec entry committed (the existing layer_drop bridge is the init builder)
- [ ] Run launched detached with retry watchdog; one watcher
- [ ] Full-set verdict with intervals recorded in RESULTS.md
- [ ] Decision: lite tier replaced or negative recorded

## Verdict (2026-09-12): NEGATIVE — 7.1402 vs run-009's 5.78

Same recipe, same canonical labels (sha e70ce991), same teacher
(re-scores 2.2921 in-run); the single variable - layer-drop init from
the trained 2.1 instead of generic byt5-small - made the student
1.36pp WORSE. Reading: generic pretraining keeps layers redundant;
task adaptation co-specializes them, and deleting half a co-adapted
stack breaks more computation. The depth axis now reads 2-of-3
negative (Hebrew generic-init collapse; Arabic adapted-init collapse;
run-009's generic-init 5.78 stands). The lite tier is unchanged; the
remaining architecture lever is /10 (lexical memory), still gated.

## Status

- [x] Build init on volume (the layer_drop bridge, student_init=2.1)
- [x] Launch (checkpoint-resumed once; canonical labels seeded after
      catching the re-labeling hazard)
- [x] Verdict: negative, recorded in RESULTS.md
