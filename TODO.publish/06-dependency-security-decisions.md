# 06 — Dependency-security decisions (the 56 Dependabot alerts)

Status: ANALYZED 2026-09-03 — decision memo for the owner. Everything
below is owner territory (version/requirements contract); nothing was
executed. Evidence gathered; PRs #51/#52/#53 are the live decision
vehicles.

## Where the alerts live

All 56 (4 critical, 20 high, 18 moderate, 14 low) target two FROZEN
legacy inference environments + the root floors:

- `python/arabic/requirements.txt` and `python/hebrew/requirements.txt`
  — 2021-era exact pins: torch==1.9.0 (2 CRITICAL advisories incl.
  the <=1.13.0 one), onnx==1.9.0, numpy==1.19.5, tqdm==4.56.0
- root `pyproject.toml` floors: torch>=2.4,<3 / onnx>=1.17 /
  onnxruntime>=1.20 / numpy>=1.26,<3 / tqdm>=4.66

interscript-ml and interscript.org have ZERO open alerts. This is a
rababa-only decision.

## The constraint (why the pins are frozen)

`python/pyproject.toml` comment: torch/onnxruntime are version-coupled
to the trained model weights; bumping requires re-validating
inference. CONFIRMED LIVE: dependabot PRs #52/#53 (torch 1.9.0 ->
2.13.0) fail their Ruby matrix with golden-output mismatches
(`expect(diacritizer.diacritize_text(source)).to eq target`) — the
legacy CBHG weights diacritize DIFFERENTLY under torch 2.13. That is
the documented re-validation gate firing, not CI flake.

Counter-evidence: `python-arabic.yml` installs with
`--upgrade-strategy eager` and its latest run is GREEN — the PYTHON
diacritize path validates fine at latest torch; only the Ruby-bridge
golden strings are numerics-pinned to the 1.9 era.

## The decision (owner picks one)

| Option | Clears alerts | Cost |
|---|---|---|
| A. Keep frozen + dismiss the legacy-file alerts (reason: vulnerable_code_not_in_use — eager CI is the live validator; the == pins are provenance) | alerts dismissed, not cleared | none; alert noise gone after dismissal (owner act via UI/API) |
| B. Merge #51/#52/#53 (torch 2.13.0 / onnx 1.22.0) + re-record the Ruby goldens at the new numerics | all legacy-file alerts | published behavior of legacy models changes; every consumer golden shifts |
| C. Raise root floors only (torch>=2.6, onnx>=1.22, tqdm>=4.66.3) | root-floor alerts | cuts off torch 2.4/2.5 consumers; requirements change |

My read: **A** for the legacy files (matches the repo's own
eager-CI validation posture), **C's floors only if** no consumer pins
rababa against torch<2.6 — that's a usage question only you can
answer. B should not be taken casually: it silently redefines what
the legacy models output.

## Fixed alongside (not owner territory)

- python/pyproject.toml pointed at TODO.complete/19-torch-2x.md which
  does not exist — pointer corrected to this memo.
