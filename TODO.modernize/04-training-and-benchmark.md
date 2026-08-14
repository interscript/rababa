# Phase 3 — Full Arabic + Hebrew training + benchmark plan

End-to-end playbook for taking rababa_arabic and rababa_hebrew from
zero to release, with a quantified "must not regress" benchmark at
the end. Each stage lists the actual commands to run.

## Stages at a glance

| Stage                  | Arabic        | Hebrew               | Compute (A100) |
|------------------------|---------------|----------------------|----------------|
| 1. Data acquisition    | ✅ in repo    | ⏳ fetch from Dicta  | —              |
| 2. Encoder + constants | ✅ exists     | ⏳ build             | —              |
| 3. MLM pretrain        | ✅ config     | ⏳ config            | ~6h each       |
| 4. Supervised fine-tune| ✅ config     | ⏳ config            | ~3h each       |
| 5. ONNX + int8 export  | ✅ code       | ⏳ multi-head export | ~30m (A10G)    |
| 6. Benchmark vs legacy | ✅ script     | ✅ script            | ~10m CPU       |
| 7. Release             | tag + ship    | tag + ship           | —              |

## Baselines (measured)

Run on Tashkeela test split (2,496 examples) via
`src/rababa/benchmark.py`:

| Model                          | DER     | Per-ex acc | Size   |
|--------------------------------|---------|------------|--------|
| Legacy 2021 Arabic (CBHG)      | **4.52%** | 8.85%    | 60 MB  |
| New v0.1.0 Arabic (target)     | ≤ 4.0%  | ≥ 10%      | ≤ 25MB |

Hebrew baseline pending (need test set — see Stage 1 below).

---

# Arabic pipeline (rababa_arabic v0.1.0)

## Stage 1 — Data ✅

`test-datasets/tashkeela/{train,val,test}.txt` already in repo.
- train: 50K lines, val/test: 2.5K each
- Format: one fully-diacritized Arabic line per row

Verify on Modal:
```bash
modal run modal_app.py::fetch_data --task rababa_arabic
```

## Stage 2 — Encoder + constants ✅

`src/rababa/constants.py` (Arabic alphabet + 15 haraqat) and
`src/rababa/encoder.py::ArabicEncoder` are ported from the legacy
2021 model. Encoder IDs are byte-identical to the 2021 trained
model — that's why the benchmark harness works against the legacy
ONNX without remapping.

## Stage 3 — MLM pretrain (~6h A100)

```bash
modal run modal_app.py::pretrain --task rababa_arabic_pretrain
# → /checkpoints/rababa_arabic_pretrain/run-001/best.pt
```

Config: `configs/rababa_arabic_pretrain.yaml` (3 epochs, batch 64,
lr 5e-4, mask_prob 0.15). Corpus: undiacritzed Tashkeela train
strip (50K lines).

## Stage 4 — Supervised fine-tune (~3h A100)

```bash
modal run modal_app.py::train \
  --task rababa_arabic \
  --init-from-pretrain /checkpoints/rababa_arabic_pretrain/run-001/best.pt
# → /checkpoints/rababa_arabic/run-001/best.pt
```

## Stage 5 — Export to ONNX + int8 (~30m A10G)

```bash
modal run modal_app.py::export_onnx \
  --task rababa_arabic \
  --version v0.1.0
# → /models/rababa_arabic/rababa_arabic-v0.1.0-fp32.onnx
# → /models/rababa_arabic/rababa_arabic-v0.1.0-q8.onnx
```

Pull artifacts locally:
```bash
modal volume get rababa-models /models/rababa_arabic/ .
```

## Stage 6 — Benchmark vs legacy

Run on the same machine (no GPU needed):

```bash
# Baseline (sanity check the number from baseline-arabic.json)
PYTHONPATH=src python -m rababa.benchmark \
  --onnx models-data/arabic-model.onnx \
  --output benchmark-legacy-arabic.json

# New model
PYTHONPATH=src python -m rababa.benchmark \
  --onnx models/rababa_arabic-v0.1.0-q8.onnx \
  --output benchmark-v0.1.0-arabic.json
```

**Acceptance gate:**
- `der` for v0.1.0 ≤ `der` for legacy (4.52%)
- ideally ≤ 4.0% (clear win, not just parity)
- per_example_accuracy ≥ legacy (8.85%)
- ONNX size ≤ 25 MB

If new model regresses → block release; investigate (probably need
more MLM epochs, or tier-2 distillation — see
`02a-mlm-pretrain.md` § "Path to v0.5.0").

## Stage 7 — Release

1. Update `ml-models/npm/models/manifest.json`: bump rababa_arabic
   to `0.1.0`, status `stable`.
2. Upload artifacts to GitHub Release `rababa_arabic-v0.1.0`:
   - `rababa_arabic-v0.1.0-q8.onnx`
   - `rababa_arabic-v0.1.0-fp32.onnx` (optional)
   - SHA256SUMS
3. Update TS runtime `DEFAULT_RABABA_CONFIGS["v0.1"]` URL.
4. Bump Ruby `rababa` gem to v0.4.0; update `Interscript.rababa_configs["v0.1"]`.
5. Smoke-test on the ISC web app end-to-end.

---

# Hebrew pipeline (rababa_hebrew v0.1.0)

Hebrew is structurally similar to Arabic (per-character
classification) but with three differences:

1. **Different alphabet + niqqud** (Hebrew letters + ~14 vowel marks
   + dagesh + sin/shin dots).
2. **Multi-head output**. The Nakdimon architecture splits the
   prediction into `niqqud` (16), `dagesh` (3), `sin` (4) — three
   independent softmaxes per position. We'll keep this design rather
   than collapsing to a single vocab, because dagesh and sin carry
   independent linguistic signal and a unified 192-class softmax
   would mostly predict "no dagesh, no sin, niqqud=X".
3. **Different test corpus**. Legacy Hebrew ONNX is Nakdimon-based;
   we need its test split for an apples-to-apples benchmark.

## Stage 1 — Data acquisition

### 1a. Modern Hebrew Nakdimon corpus (Dicta)

Source: `https://github.com/elazarg/nakdimon` — Nakdimon's training
data (Modern Hebrew, ~500K sentences with nikud). Licensed MIT.

```bash
# Inside the Modal fetch_data function (to be extended for Hebrew)
git clone https://github.com/elazarg/nakdimon.git /tmp/nakdimon
# The corpus lives at /tmp/nakdimon/data/{train,val,test}.txt
```

Expected split sizes:
- train: ~470K lines
- val: ~2K
- test: ~2K

### 1b. Sample for parity with Arabic pipeline

Subsample train to ~50K lines (matches Tashkeela scale; keeps Modal
compute equal). Keep val/test at full ~2K each.

### 1c. Verify on Modal

Add a `rababa_hebrew` branch to `modal_app.py::fetch_data` that
fetches + checks Nakdimon corpus SHA256s.

## Stage 2 — Encoder + constants (NEW CODE)

### 2a. `src/rababa/constants_hebrew.py`

Port from `lib/rababa/hebrew.rb` (already defines the alphabet):

```python
HEBREW_LETTERS = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט",
                  "י", "ך", "כ", "ל", "ם", "מ", "ן", "נ", "ס",
                  "ע", "ף", "פ", "ץ", "צ", "ק", "ר", "ש", "ת"]

NIQQUD = {  # 14 values
    "": "None", "ְ": "Shva", "ֱ": "Reduced Segol", ...
}
DAGESH = {"": "None", "ּ": "Dagesh", ...}  # 3 values
SIN = {"": "None", "ׁ": "Sin", "ׂ": "Shin", ...}  # 4 values
```

### 2b. `src/rababa/encoder.py::HebrewEncoder`

Mirror `ArabicEncoder` with Hebrew cleaner. Same encode/clean/decode API.

### 2c. `src/rababa/datasets.py::NakdimonDataset`

Mirror `TashkeelaDataset`. Each line yields:
- `input_ids`: undiacritized Hebrew letter IDs
- `target_niqqud_ids`, `target_dagesh_ids`, `target_sin_ids`: per-position targets

### 2d. `src/rababa/models/student.py::MultiHeadCharTransformer`

Same `CharTransformer` body, three `nn.Linear` heads. Or: subclass
`CharTransformer`, override `forward` to return a tuple/dict of
logits. OCP-compliant: existing single-head student untouched.

## Stage 3 — MLM pretrain (~6h A100)

Same recipe as Arabic. Hebrew alphabet → similar vocab size, same
architecture works.

Add `configs/rababa_hebrew_pretrain.yaml` (mirror
`rababa_arabic_pretrain.yaml`).

```bash
modal run modal_app.py::pretrain --task rababa_hebrew_pretrain
```

## Stage 4 — Supervised fine-tune (~3h A100)

`configs/rababa_hebrew.yaml` — same hyperparams as Arabic, target
DER ≤ 12% (looser than Arabic since Modern Hebrew nikud is harder).

Loss = niqqud_loss + dagesh_loss + sin_loss (simple sum; can weight later).

```bash
modal run modal_app.py::train \
  --task rababa_hebrew \
  --init-from-pretrain /checkpoints/rababa_hebrew_pretrain/run-001/best.pt
```

## Stage 5 — Export to ONNX + int8 (~30m A10G)

Multi-head export: 3 outputs instead of 1. The legacy Hebrew ONNX
contract is:
- input: `normalized` [32, dyn]
- outputs: `niqqud` [32, dyn, 16], `dagesh` [32, dyn, 3], `sin` [32, dyn, 4]

Match this contract so the Ruby/TS runtime needs no change. Add a
`MultiHeadExporter` parallel to `export_student_onnx`.

```bash
modal run modal_app.py::export_onnx --task rababa_hebrew --version v0.1.0
```

## Stage 6 — Benchmark vs legacy

Need to extend `benchmark.py` to handle multi-head models. Algorithm:
- Run legacy ONNX → get (niqqud, dagesh, sin) per position
- Run new ONNX → get (niqqud, dagesh, sin) per position
- Decode both to actual Hebrew text with niqqud+dagesh+sin applied
- Compute **text-level DER**: fraction of positions where decoded char ≠ gold char

This is fairer than per-head DER because it measures what the user
sees. Single-head DER is reported as a secondary metric.

```bash
PYTHONPATH=src python -m rababa.benchmark \
  --onnx models-data/hebrew-model.onnx \
  --output benchmark-legacy-hebrew.json

PYTHONPATH=src python -m rababa.benchmark \
  --onnx models/rababa_hebrew-v0.1.0-q8.onnx \
  --output benchmark-v0.1.0-hebrew.json
```

**Acceptance gate:** new Hebrew model DER ≤ legacy Hebrew DER on the
same Nakdimon test split.

## Stage 7 — Release

Same template as Arabic Stage 7:
1. Update `ml-models` manifest: `rababa_hebrew` 0.1.0, status `stable`.
2. Upload artifacts to GH release `rababa_hebrew-v0.1.0`.
3. Update TS `setRababaConfig("hebrew-v0.1", ...)`.
4. Bump Ruby gem to v0.4.0; update `Interscript.rababa_configs["hebrew-v0.1"]`.

---

# Cross-cutting concerns

## Reproducibility

Every benchmark result file (`benchmark-{tag}-{lang}.json`) records:
- model path + size
- task + split + cleaner
- n_examples + n_batches
- DER + per-example accuracy
- I/O contract (input/output shapes)

Commit each result file alongside the model release. This is the
artifact a future maintainer uses to verify "we didn't regress".

## Modal cost budget

| Run                          | GPU    | Wall time | Cost (paid Modal) |
|------------------------------|--------|-----------|-------------------|
| Arabic pretrain              | A100   | 6h        | ~$12              |
| Arabic fine-tune             | A100   | 3h        | ~$6               |
| Hebrew pretrain              | A100   | 6h        | ~$12              |
| Hebrew fine-tune             | A100   | 3h        | ~$6               |
| Arabic + Hebrew export       | A10G   | 1h total  | ~$1               |
| **Total v0.1.0 (both langs)** |        | **~19h**  | **~$37**          |

Tier 2 distillation (if triggered) adds ~$20-40 per language.

## Failure modes + recovery

| Symptom                                   | Diagnosis                         | Fix                                   |
|-------------------------------------------|-----------------------------------|---------------------------------------|
| New model DER > legacy                    | Undertrained encoder              | Add MLM epochs; check pretrain loss   |
| New model DER much worse (e.g. 30%+)      | Vocab mismatch                    | Verify encoder IDs match what ONNX expects |
| ONNX export fails on shape                | dynamic_axes conflict             | Re-export with fully fixed shape      |
| int8 quantization degrades DER > 1pt      | Calibration set too small         | Re-quantize with 5K-sample calib set  |
| Ruby adapter crashes on new model         | I/O contract drift                | Verify input names unchanged          |

## Rollback path

If v0.1.0 ships and a regression is discovered in production:

1. Bump manifest version pointer back to `0.0.0` (= legacy model URL).
2. Redeploy TS/Ruby.
3. Investigate via `benchmark.py` + a fresh test split.

The legacy ONNX models stay in `models-data/` indefinitely — they
are the rollback target.
