"""The Sadeed windowed harness — the campaign's central measurement
protocol, in one place.

Windowed zero-skip (the published protocol): inputs over the byte
budget split at word boundaries; greedy decode with a 2x-window
generation cap (diacritized output runs 1.4-1.6x input — a shorter cap
silently truncates the hardest paragraphs, which the evaluator then
skips: survivorship bias, not quality); window predictions stitched and
haraqat projected onto the input letters so output structure always
matches ground truth.

Previously pasted verbatim into rababa train_arabic_r5/r6/r7/r8 and
modal_distill.evaluate_der — five copies of a protocol every published
Arabic number depends on. rababa scripts should vendor this single
module instead of carrying inline copies.
"""

from __future__ import annotations

import re

DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")


def split_windows(text: str, budget: int = 1400) -> list[str]:
    """Split at word boundaries so no window exceeds the byte budget."""
    if len(text.encode("utf-8")) <= budget:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    n = 0
    for word in text.split():
        cost = len(word.encode("utf-8")) + 1
        if current and n + cost > budget:
            windows.append(" ".join(current))
            current, n = [], 0
        current.append(word)
        n += cost
    if current:
        windows.append(" ".join(current))
    return windows


def project_haraqat(pred: str, text: str) -> str:
    """Project predicted haraqat onto the input's letters, so the output
    structure matches ground truth even when the prediction inserts or
    drops letters (zero-skip contract)."""
    from difflib import SequenceMatcher

    pred_haraqat = [""]
    for ch in pred:
        if DIACRITICS_RE.match(ch):
            pred_haraqat[-1] += ch
        else:
            pred_haraqat.append("")
    pred_haraqat = pred_haraqat[1:]
    pred_letters = [c for c in pred if not DIACRITICS_RE.match(c)]
    text_letters = [c for c in text if not DIACRITICS_RE.match(c)]
    sm = SequenceMatcher(None, text_letters, pred_letters, autojunk=False)
    out: list[str] = []
    for op, i1, i2, j1, _j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out.append(text_letters[i1 + k] + pred_haraqat[j1 + k])
        else:
            for k in range(i1, i2):
                out.append(text_letters[k])
    return "".join(out)


def strip_diacritics(text: str) -> str:
    return DIACRITICS_RE.sub("", text)


def windowed_paragraphs(model, tokenizer, inputs, window: int = 1400,
                        batch_size: int = 8, device: str = "cuda") -> list[str]:
    """The full protocol over a paragraph list: split, greedy-decode each
    window (2x-window cap) under bf16 autocast, stitch, project. Returns
    haraqat-projected paragraphs ready for the Misraj evaluator."""
    import torch

    windows: list[str] = []
    counts: list[int] = []
    for text in inputs:
        ws = split_windows(text, window)
        counts.append(len(ws))
        windows.extend(ws)

    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=window,
            ).to(device)
            with torch.autocast("cuda", torch.bfloat16):
                gen = model.generate(**enc, max_new_tokens=window * 2, num_beams=1)
            preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

    paragraphs: list[str] = []
    k = 0
    for text, c in zip(inputs, counts, strict=True):
        paragraphs.append(project_haraqat(" ".join(preds[k : k + c]), text))
        k += c
    return paragraphs
