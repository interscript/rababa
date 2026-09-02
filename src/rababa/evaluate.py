"""Evaluation metrics — Diacritization Error Rate (DER) and PER.

DER: per-character error rate. For each example, count the positions
where the predicted haraqat ID != target ID (ignoring pad positions).
Average over all examples, weighted by length.

PER: per-example error rate. Binary — an example is "correct" only if
ALL haraqat positions match.

All metrics accept an optional `lexicon` parameter. When provided,
predictions are re-decoded per-word using `trie_constrained_decode`
(forcing valid haraqat sequences per word from the lexicon). This is
an inference-time-only quality boost — zero retraining required.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .constants import PAD_ID


def _flatten_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    src: torch.Tensor | None = None,
    lexicon: dict[str, list[list[int]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop PAD positions, return (predictions, targets) 1D tensors.

    If `lexicon` is provided alongside `src`, predictions are produced
    via trie-constrained per-word decoding instead of argmax.
    """
    if lexicon is not None and src is not None:
        from .decoding.constrained import trie_constrained_decode
        predictions = trie_constrained_decode(logits, src, lexicon)
    else:
        predictions = logits.argmax(dim=-1)
    mask = target != PAD_ID
    return predictions[mask], target[mask]


def diacritization_error_rate(
    logits: torch.Tensor,
    target: torch.Tensor,
    src: torch.Tensor | None = None,
    lexicon: dict[str, list[list[int]]] | None = None,
) -> float:
    """Per-character DER — fraction of haraqat positions predicted wrong.

    Lower is better. 0.0 = perfect.

    Pass `src` + `lexicon` to enable trie-constrained decoding.
    """
    preds, targets = _flatten_logits(logits, target, src, lexicon)
    if targets.numel() == 0:
        return 0.0
    wrong = (preds != targets).sum().item()
    return wrong / targets.numel()


def per_example_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    src: torch.Tensor | None = None,
    lexicon: dict[str, list[list[int]]] | None = None,
) -> float:
    """Fraction of examples where ALL haraqat positions are correct.

    Higher is better. 1.0 = perfect.

    Pass `src` + `lexicon` to enable trie-constrained decoding.
    """
    if lexicon is not None and src is not None:
        from .decoding.constrained import trie_constrained_decode
        predictions = trie_constrained_decode(logits, src, lexicon)
    else:
        predictions = logits.argmax(dim=-1)
    mask = target != PAD_ID
    batch_size = target.size(0)
    correct = 0
    for i in range(batch_size):
        if torch.equal(predictions[i][mask[i]], target[i][mask[i]]):
            correct += 1
    return correct / max(1, batch_size)


def compute_der_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Alias for `diacritization_error_rate` (no lexicon)."""
    return diacritization_error_rate(logits, target)


def compute_der(predictions: list[int], targets: list[int]) -> float:
    """DER from flat integer lists (for ONNX parity tests)."""
    if not targets:
        return 0.0
    wrong = sum(1 for p, t in zip(predictions, targets, strict=False) if p != t)
    return wrong / len(targets)


# ---- Seq2seq DER ----

_NIQQUD_MARKS = set("ְֱֲֳִֵֶַָֹֺֻּֽֿׁׂ־")


def seq2seq_der(generated: str, gold: str) -> tuple[float, int]:
    """DER for seq2seq Hebrew diacritization.

    Parses both texts into (consonant, niqqud) units, aligns by consonant,
    and counts mismatches. Missing or extra niqqud counts as an error.

    Returns (der, total_positions).
    """
    def _parse(text):
        units = []
        current_consonant = None
        current_niqqud = []
        for ch in text:
            if ch in _NIQQUD_MARKS:
                current_niqqud.append(ch)
            else:
                if current_consonant is not None:
                    units.append((current_consonant, "".join(current_niqqud)))
                current_consonant = ch
                current_niqqud = []
        if current_consonant is not None:
            units.append((current_consonant, "".join(current_niqqud)))
        return units

    gen_units = _parse(generated)
    gold_units = _parse(gold)

    if not gold_units:
        return 0.0, 0

    # Align by consonant. If skeletons differ, every mismatched position is an error.
    wrong = 0
    total = len(gold_units)
    for i in range(total):
        if i >= len(gen_units):
            wrong += 1
            continue
        g_cons, g_niq = gen_units[i]
        c_cons, c_niq = gold_units[i]
        if g_cons != c_cons or g_niq != c_niq:
            wrong += 1

    der = wrong / total
    return der, total


def seq2seq_batch_der(
    model,
    src: torch.Tensor,
    src_kpm: torch.Tensor,
    id_to_char: list[str],
    gold_texts: list[str],
    device: torch.device,
    max_steps: int = 600,
    copy_augmented: bool = True,
) -> tuple[float, int]:
    """Decode a batch and compute aggregate DER against gold texts.

    Args:
        model: HebrewSeq2Seq model.
        src: (B, T_src) undiacritized char IDs.
        src_kpm: (B, T_src) source key padding mask.
        id_to_char: vocab list (index → char).
        gold_texts: list of gold diacritized strings.
        device: torch device.
        max_steps: max decoder steps.
        copy_augmented: if True, force consonant tokens to match the input
            (only niqqud marks are model-generated). This eliminates the
            consonant copy problem caused by exposure bias.

    Returns (aggregate_der, total_positions).
    """
    memory, _ = model.encode(src)
    B = src.size(0)
    T_src = src.size(1)
    tgt = torch.full((B, 1), model.BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    total_wrong = 0
    total_positions = 0

    # Track input consonant pointer for copy-augmented decoding.
    consonant_ptr = torch.zeros(B, dtype=torch.long, device=device)

    for _ in range(max_steps):
        if finished.all():
            break
        x = model.embedding(tgt)
        cos, sin = model.rotary(tgt.size(1))
        for layer in model.decoder_layers:
            x = layer(x, memory, cos, sin, None, src_kpm)
        x = model.dec_norm(x)
        logits = model.head(x[:, -1:])
        next_token = logits.argmax(dim=-1)  # (B, 1)

        if copy_augmented:
            for i in range(B):
                if finished[i]:
                    next_token[i] = model.EOS_ID
                    continue
                tid = next_token[i].item()
                ch = id_to_char[tid] if 0 <= tid < len(id_to_char) else ""
                if ch in _NIQQUD_MARKS:
                    pass  # model-generated niqqud — keep it
                else:
                    ptr = consonant_ptr[i].item()
                    while ptr < T_src and src_kpm[i, ptr]:
                        ptr += 1
                    if ptr < T_src:
                        next_token[i] = src[i, ptr]
                        consonant_ptr[i] = ptr + 1
                    else:
                        next_token[i] = model.EOS_ID
                        finished[i] = True

        eos_mask = next_token.squeeze(-1) == model.EOS_ID
        finished = finished | eos_mask
        next_token = next_token.masked_fill(finished.unsqueeze(1), model.EOS_ID)
        tgt = torch.cat([tgt, next_token], dim=1)

    generated_texts = []
    for i in range(B):
        ids = tgt[i].tolist()
        text = []
        for tid in ids[1:]:  # skip BOS
            if tid == model.EOS_ID:
                break
            if 0 <= tid < len(id_to_char):
                text.append(id_to_char[tid])
        generated_texts.append("".join(text))

    for gen, gold in zip(generated_texts, gold_texts, strict=True):
        der, n = seq2seq_der(gen, gold)
        total_wrong += int(der * n)
        total_positions += n

    return total_wrong / max(1, total_positions), total_positions


def load_lexicon_for_eval(path: str | Path | None) -> dict[str, list[list[int]]] | None:
    """Load a lexicon JSON, or return None if path is None / missing.

    Convenience wrapper for eval callers that take an optional lexicon path.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    import json
    return json.loads(p.read_text(encoding="utf-8"))
