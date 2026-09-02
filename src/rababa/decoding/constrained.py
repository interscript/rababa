"""Trie-constrained decoder — exact per-word search over the lexicon.

For each word in the input:
  - In-vocab: enumerate all haraqat sequences, score each by sum of
    per-char log-probs from the model, pick highest. Exact, no beam
    approximation needed (per-word enumeration is small — typically
    1-5 sequences after top-K pruning).
  - OOV: per-character argmax (the lexicon can't help).

Returns a (B, T) tensor of haraqat IDs, same shape as model.argmax(-1)
but with in-vocab words decoded optimally given the constraint.
"""

from __future__ import annotations

import math

import torch

from ..constants import HARAQAT, PAD_ID


# ---- Word segmentation ------------------------------------------------


def find_word_spans(src: torch.Tensor, pad_id: int = PAD_ID) -> list[list[tuple[int, int]]]:
    """Return per-example word spans as (start, end) char indices.

    Words are delimited by whitespace (char ID for space) and PAD. Each
    span is half-open: [start, end). Empty spans (length 0) are skipped.
    """
    # Look up the space char ID once (it's in ARAB_CHARS).
    space_id = _space_char_id()
    spans: list[list[tuple[int, int]]] = []
    for row in src:
        row_spans: list[tuple[int, int]] = []
        start = 0
        for i, c in enumerate(row.tolist()):
            if c == pad_id or c == space_id:
                if i > start:
                    row_spans.append((start, i))
                start = i + 1
        # last word
        last = len(row)
        if last > start:
            row_spans.append((start, last))
        spans.append(row_spans)
    return spans


_SPACE_ID_CACHE: int | None = None


def _space_char_id() -> int:
    global _SPACE_ID_CACHE
    if _SPACE_ID_CACHE is not None:
        return _SPACE_ID_CACHE
    from ..constants import VALID_ARABIC
    try:
        _SPACE_ID_CACHE = VALID_ARABIC.index(" ") + 1  # +1 for PAD at index 0
    except ValueError:
        _SPACE_ID_CACHE = 0
    return _SPACE_ID_CACHE


# ---- Per-word scoring -------------------------------------------------


def _score_sequence(word_logits: torch.Tensor, sequence: list[int]) -> float:
    """Sum of log-softmax probs along `sequence`. Higher = better."""
    # log_softmax along vocab axis (numerically stable, vocab is small ~17)
    log_probs = torch.log_softmax(word_logits, dim=-1)
    score = 0.0
    for i, target_id in enumerate(sequence):
        if i >= log_probs.shape[0]:
            return -math.inf  # sequence longer than word — invalid
        score += log_probs[i, target_id].item()
    return score


def _decode_word(
    word_logits: torch.Tensor,
    candidates: list[list[int]],
) -> list[int]:
    """Pick the best haraqat sequence for one word."""
    word_len = word_logits.shape[0]
    if not candidates:
        return word_logits.argmax(dim=-1).tolist()

    # Filter candidates by length mismatch (can't apply if lengths differ).
    valid_candidates = [c for c in candidates if len(c) == word_len]
    if not valid_candidates:
        return word_logits.argmax(dim=-1).tolist()

    best_seq: list[int] | None = None
    best_score = -math.inf
    for cand in valid_candidates:
        s = _score_sequence(word_logits, cand)
        if s > best_score:
            best_score = s
            best_seq = cand
    return best_seq if best_seq is not None else word_logits.argmax(dim=-1).tolist()


# ---- Batch decoder ----------------------------------------------------


def trie_constrained_decode(
    logits: torch.Tensor,
    src: torch.Tensor,
    lexicon: dict[str, list[list[int]]],
    undiacritized_words: list[list[str]] | None = None,
) -> torch.Tensor:
    """Apply lexicon constraint to per-char haraqat predictions.

    Args:
        logits: (B, T, V) model output logits.
        src:    (B, T) input char IDs (used for word segmentation).
        lexicon: {undiacritized_word: [[haraqat_ids...], ...]}
        undiacritized_words: optional per-example list of words aligned to
            the auto-detected word spans. If None, words are looked up by
            re-decoding the source — caller typically passes this.

    Returns:
        (B, T) long tensor of haraqat IDs.
    """
    B, T, _ = logits.shape
    out = logits.argmax(dim=-1).long()  # default: per-char argmax

    spans_per_example = find_word_spans(src)
    for b in range(B):
        spans = spans_per_example[b]
        words = (undiacritized_words[b] if undiacritized_words is not None
                 else _reconstruct_words(src[b], spans))
        for (start, end), word in zip(spans, words, strict=False):
            if not word:
                continue
            candidates = lexicon.get(word)
            if not candidates:
                continue  # OOV — keep argmax
            word_logits = logits[b, start:end, :]
            best = _decode_word(word_logits, candidates)
            out[b, start:end] = torch.tensor(best, dtype=out.dtype, device=out.device)
    return out


def _reconstruct_words(src_row: torch.Tensor, spans: list[tuple[int, int]]) -> list[str]:
    """Decode source IDs back to strings for lexicon lookup.

    Inverse of the encoder. The space char and PAD are excluded from spans
    by `find_word_spans`, so each span maps cleanly to one word's chars.
    """
    from ..constants import VALID_ARABIC
    # Build ID → char table. PAD at 0, then VALID_ARABIC[0..].
    id_to_char = [""] + VALID_ARABIC
    out: list[str] = []
    for start, end in spans:
        chars = []
        for cid in src_row[start:end].tolist():
            if 0 <= cid < len(id_to_char):
                chars.append(id_to_char[cid])
            else:
                chars.append("")
        out.append("".join(chars))
    return out


def apply_lexicon_to_batch(
    logits: torch.Tensor,
    src: torch.Tensor,
    lexicon_path: str | None,
    undiacritized_words: list[list[str]] | None = None,
) -> torch.Tensor:
    """Convenience wrapper: load lexicon from path if given, else argmax."""
    if lexicon_path is None:
        return logits.argmax(dim=-1).long()
    from pathlib import Path
    from .lexicon import load_lexicon
    lex = load_lexicon(Path(lexicon_path))
    return trie_constrained_decode(logits, src, lex, undiacritized_words)
