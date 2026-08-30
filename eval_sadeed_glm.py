"""Evaluate a GLM model (z.ai API) on SadeedDiac-25 — clean protocol.

Same benchmark, same evaluator as every row in our table: undiacritized
input paragraph -> fully diacritized output, temperature 0, word/letter
structure preserved. Reports both the raw protocol (as the published
LLM rows used) and the projected zero-skip variant.

Checkpoints every response to CKPT so the run resumes after
interruption. Key read from ~/.zai-api-key (never printed).

reasoning_effort is required for glm-5.3+ models: absent or
unrecognized values silently default to MAX reasoning there — the
2026-08-17 finding was that reasoning mode burns minutes per long
paragraph, so a missing parameter turns a ~1h run into days. GLM-5.2
and earlier disable reasoning via thinking.type=disabled; GLM-5.3+
rejects that parameter outright (verified 2026-08-31: HTTP 400 code
1210, valid reasoning_effort values are exactly low, high, max —
thinking cannot be disabled at all). An effort therefore sends ONLY
reasoning_effort, and any response still carrying reasoning_content
warns that it did not take.

Usage:
    python eval_sadeed_glm.py [model_id] [reasoning_effort]
    # default glm-5.2; glm-5.3* refuses to run without an effort value
    # e.g. python eval_sadeed_glm.py glm-5.3-flash low
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "glm-5.2"
EFFORT = sys.argv[2] if len(sys.argv) > 2 else None
if MODEL.startswith("glm-5.3") and not EFFORT:
    sys.exit(
        f"refusing to run {MODEL}: absent/unrecognized reasoning_effort "
        "silently defaults to MAX (minutes per paragraph). Pass an effort "
        f"value, e.g.: python eval_sadeed_glm.py {MODEL} low")
DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
CKPT = Path(f"/tmp/sadeed_glm_{MODEL.replace('.', '_')}"
            + (f"_{EFFORT}" if EFFORT else "") + ".jsonl")
OUT_DIR = Path("results") / f"sadeed-{MODEL.replace('.', '-')}"
BASE = "https://api.z.ai/api/paas/v4/chat/completions"

PROMPT = """Add complete diacritization (tashkeel: fatha, damma, kasra, sukun, shadda, tanween, and case endings) to every word of the Arabic text below. Return ONLY the fully diacritized Arabic text — no explanations, no translation, no markdown. Preserve every letter, word, number, punctuation mark and line exactly as given.

Text: {t}"""


def load_data() -> tuple[list[str], list[str]]:
    table = pq.read_table("data/sadeed-diac-25/train.parquet")
    inputs = [DIAC.sub("", t) for t in table.column("input").to_pylist()]
    outputs = table.column("output").to_pylist()
    return inputs, outputs


def clean(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```[a-z]*\n?|\n?```$", "", txt).strip()
    return txt


def call(session: requests.Session, key: str, text: str, tries: int = 5) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(t=text)}],
        "temperature": 0,
        "max_tokens": 8192,
    }
    if EFFORT:
        # glm-5.3+ rejects thinking.type=disabled outright (HTTP 400,
        # code 1210: "This model always engages in thinking"); valid
        # reasoning_effort values are low, high, max.
        payload["reasoning_effort"] = EFFORT
    else:
        payload["thinking"] = {"type": "disabled"}
    for attempt in range(tries):
        try:
            r = session.post(BASE, json=payload, timeout=300)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            message = r.json()["choices"][0]["message"]
            if message.get("reasoning_content"):
                print(f"[glm] WARNING: reasoning_content present — "
                      f"reasoning disable not honored by {MODEL}", flush=True)
            return clean(message["content"])
        except requests.RequestException:
            if attempt == tries - 1:
                return ""
            time.sleep(5 * (attempt + 1))
    return ""


def main() -> None:
    key = open(os.path.expanduser("~/.zai-api-key")).read().strip()
    inputs, outputs = load_data()

    done: dict[int, str] = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["idx"]] = row["pred"]
    todo = [i for i in range(len(inputs)) if i not in done]
    print(f"[glm] model={MODEL} effort={EFFORT or 'thinking-disabled'} "
          f"done={len(done)} todo={len(todo)}", flush=True)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    with CKPT.open("a", encoding="utf-8") as ckpt:
        def work(i: int) -> tuple[int, str]:
            return i, call(session, key, inputs[i])

        n_written = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(work, i): i for i in todo}
            for fut in as_completed(futures):
                i, pred = fut.result()
                done[i] = pred
                ckpt.write(json.dumps({"idx": i, "pred": pred}, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % 20 == 0:
                    ckpt.flush()
                    print(f"[glm] {len(done)}/{len(inputs)}", flush=True)

    preds = [done[i] for i in range(len(inputs))]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame({"gt": outputs, "pred": preds}).to_csv(
        OUT_DIR / "sadeed_preds_raw.csv", index=False, header=False)
    empty = sum(1 for p in preds if not p)
    print(f"[glm] finished, empty responses: {empty}", flush=True)

    from sadeed_evaluator import ArabicDiacritizationEvaluator as E

    print(f"\n===== {MODEL} raw protocol (their default) =====", flush=True)
    E.report_errors_on_csv_file(
        str(OUT_DIR / "sadeed_preds_raw.csv"), ground_truth_column_index=0,
        predicted_column_index=1, has_header=False, gt_missing_diacritic_is_error=False)

    from difflib import SequenceMatcher

    def project_haraqat(pred: str, text: str) -> str:
        pred_haraqat = [""]
        for ch in pred:
            if DIAC.match(ch):
                pred_haraqat[-1] += ch
            else:
                pred_haraqat.append("")
        pred_haraqat = pred_haraqat[1:]
        pred_letters = [c for c in pred if not DIAC.match(c)]
        text_letters = [c for c in text if not DIAC.match(c)]
        sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
        out = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                for k in range(i2 - i1):
                    out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
            else:
                for k in range(i1, i2):
                    out.append(text_letters[k])
        return "".join(out)

    projected = [project_haraqat(p, inp) for p, inp in zip(preds, inputs)]
    pd.DataFrame({"gt": outputs, "pred": projected}).to_csv(
        OUT_DIR / "sadeed_preds_projected.csv", index=False, header=False)
    print(f"\n===== {MODEL} projected zero-skip protocol =====", flush=True)
    E.report_errors_on_csv_file(
        str(OUT_DIR / "sadeed_preds_projected.csv"), ground_truth_column_index=0,
        predicted_column_index=1, has_header=False, gt_missing_diacritic_is_error=False)


if __name__ == "__main__":
    main()
