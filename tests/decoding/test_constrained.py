"""Specs for trie-constrained decoding + lexicon."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rababa.constants import TARGET_VOCAB, VALID_ARABIC
from rababa.decoding.constrained import (
    _decode_word,
    _score_sequence,
    find_word_spans,
    trie_constrained_decode,
)
from rababa.decoding.lexicon import Lexicon, load_lexicon, save_lexicon


# ---- Lexicon ---------------------------------------------------------


def test_lexicon_add_and_build_returns_top_k_per_word():
    lex = Lexicon(top_k_per_word=2, min_word_freq=1)
    # Same word with 3 different haraqat sequences, frequencies 3/2/1.
    lex.add("سلام", (1, 2, 3))
    lex.add("سلام", (1, 2, 3))
    lex.add("سلام", (1, 2, 3))
    lex.add("سلام", (4, 5, 6))
    lex.add("سلام", (4, 5, 6))
    lex.add("سلام", (7, 8, 9))

    data = lex.build()
    assert "سلام" in data
    # top_k=2 means only the top-2 most frequent sequences are kept.
    assert len(data["سلام"]) == 2
    # Most-frequent sequence is first.
    assert data["سلام"][0] == [1, 2, 3]


def test_lexicon_min_word_freq_filters_rare_words(tmp_path: Path):
    lex = Lexicon(top_k_per_word=5, min_word_freq=3)
    for _ in range(3):
        lex.add("کتاب", (1, 2))
    lex.add("نادر", (1,))  # only seen once — filtered out
    data = lex.build()
    assert "کتاب" in data
    assert "نادر" not in data


def test_save_and_load_lexicon_roundtrips(tmp_path: Path):
    lex = Lexicon(top_k_per_word=3, min_word_freq=1)
    lex.add("سلام", (1, 2, 3))
    lex.add("سلام", (4, 5))
    out_path = tmp_path / "lex.json"
    stats = save_lexicon(lex, out_path)
    assert stats["entries"] == 1
    loaded = load_lexicon(out_path)
    assert "سلام" in loaded
    assert loaded["سلام"] == [[1, 2, 3], [4, 5]]  # sorted by freq desc


# ---- Trie-constrained decode ----------------------------------------


def test_find_word_spans_splits_on_space_and_pad():
    # Use the actual space char ID (19) and pad at the end.
    from rababa.decoding.constrained import _space_char_id
    space_id = _space_char_id()
    src = torch.tensor([[5, 6, 7, space_id, 8, 9, 0, 0]])  # 4 = space id placeholder
    spans = find_word_spans(src, pad_id=0)
    assert len(spans) == 1
    # Two words split by the space.
    spans_row = spans[0]
    assert len(spans_row) == 2
    assert spans_row[0] == (0, 3)  # first three positions
    assert spans_row[1] == (4, 6)  # after space to before pad


def test_score_sequence_prefers_high_log_prob():
    # Two positions, vocab 4. logits favor class 0 at pos 0, class 2 at pos 1.
    logits = torch.tensor([
        [[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0]],
    ])  # shape (1, 2, 4)
    good = _score_sequence(logits[0], [0, 2])
    bad = _score_sequence(logits[0], [1, 1])
    assert good > bad


def test_decode_word_returns_argmax_when_no_candidates():
    # Empty candidate list → fall back to argmax.
    logits = torch.tensor([[0.0, 1.0, 5.0], [3.0, 0.0, 0.0]])
    out = _decode_word(logits, [])
    assert out == logits.argmax(dim=-1).tolist()


def test_decode_word_picks_best_matching_candidate_by_length():
    # Word has length 2; only candidate of length 2 should be considered.
    logits = torch.tensor([
        [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],  # argmax = [0, 2]
    ])
    logits = logits[0]
    # Candidate of wrong length is filtered.
    candidates_wrong = [[0, 2, 9]]  # length 3
    out = _decode_word(logits, candidates_wrong)
    # Falls back to argmax since no length-matching candidate.
    assert out == [0, 2]

    # Candidate of right length, score it.
    candidates_right = [[0, 2], [1, 1]]
    out = _decode_word(logits, candidates_right)
    assert out == [0, 2]  # matches argmax since both logits favor it


def test_trie_constrained_decode_overwrites_in_vocab_words():
    # Batch of 1, seq len 4, vocab 3.
    # Model strongly predicts class 1 at every position.
    from rababa.decoding.constrained import _space_char_id
    space_id = _space_char_id()
    logits = torch.full((1, 5, 3), -10.0)
    logits[..., 1] = 10.0  # argmax = [1,1,1,1,1]

    # Source: "word1 word2" with a real space at position 2.
    # word1 = positions 0,1; word2 = positions 3,4.
    src = torch.tensor([[10, 11, space_id, 12, 13]])

    # Lexicon: word1 → [0,0] only.
    lexicon = {"ab": [[0, 0]]}
    out = trie_constrained_decode(
        logits, src, lexicon, undiacritized_words=[["ab", "cd"]],
    )
    # Positions 0-1 overwritten with 0 (lexicon); position 2 (space) and 3-4 keep argmax.
    assert out[0, 0].item() == 0
    assert out[0, 1].item() == 0
    assert out[0, 2].item() == 1  # space position keeps argmax
    assert out[0, 3].item() == 1  # word2 OOV keeps argmax
    assert out[0, 4].item() == 1


def test_trie_constrained_decode_falls_back_to_argmax_for_oov():
    logits = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ]])
    src = torch.tensor([[5, 6]])
    # Empty lexicon → all words OOV → argmax.
    out = trie_constrained_decode(logits, src, {}, undiacritized_words=[["xyz"]])
    assert out[0].tolist() == [0, 1]  # argmax
