import haraqat_planes
"""Tests for the haraqat plane decomposition (Stoicheia WO, run-017).

A diacritized Arabic string factors into two aligned planes:
    skeleton[i] = base letters (no combining marks)
    labels[i]   = canonical combining-mark combo following skeleton[i]
render(skeleton, labels) is the exact inverse.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from haraqat_planes import canon_combo, render, split_planes


class TestCanonCombo(unittest.TestCase):
    def test_shadda_sorts_before_haraka(self):
        self.assertEqual(canon_combo("َّ"), canon_combo("َّ"))
        self.assertEqual(canon_combo("َّ"), "َّ")

    def test_empty(self):
        self.assertEqual(canon_combo(""), "")


class TestSplitRender(unittest.TestCase):
    def test_basic_roundtrip(self):
        text = "كَتَبَ الْعَرَبِيَّةُ"
        skel, labels = split_planes(text)
        self.assertEqual(skel, "كتب العربية")
        self.assertEqual(labels[0], "َ")          # fatha on ك
        self.assertEqual(labels[2], "َ")          # fatha on ب
        self.assertEqual(labels[3], "")           # space
        self.assertEqual(labels[5], "ْ")          # sukun on ل
        self.assertEqual(labels[9], "َّ")         # shadda+kasra on ي
        self.assertEqual(labels[10], "ُ")         # damma on ة
        self.assertEqual(render(skel, labels), text)

    def test_plain_undiacritized(self):
        skel, labels = split_planes("كتب")
        self.assertEqual((skel, labels), ("كتب", ["", "", ""]))
        self.assertEqual(render(skel, labels), "كتب")

    def test_tanwin_and_mixed(self):
        text = "كِتَابًا مُحَمَّدٍ"
        skel, labels = split_planes(text)
        self.assertEqual(labels[3], "ً")          # fathatan on ب
        self.assertEqual(labels[8], "َّ")         # shadda+fatha on م
        self.assertEqual(labels[9], "ٍ")          # kasratan on د
        self.assertEqual(labels[5], "")           # space
        self.assertEqual(render(skel, labels), text)

    def test_non_arabic_passthrough(self):
        text = "hello كَتَبَ world 123"
        skel, labels = split_planes(text)
        self.assertEqual(skel, "hello كتب world 123")
        self.assertEqual(render(skel, labels), text)

    def test_length_invariant(self):
        text = "وَقَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"
        skel, labels = split_planes(text)
        self.assertEqual(len(skel), len(labels))
        self.assertEqual(render(skel, labels), text)


class TestRoundTripOnBenchmarkCorpus(unittest.TestCase):
    def test_sadeed_outputs_roundtrip(self):
        import pandas as pd

        df = pd.read_parquet("data/sadeed-diac-25/train.parquet")
        n_bad = 0
        for text in df["output"].head(500):
            skel, labels = split_planes(text)
            if render(skel, labels) != text:
                n_bad += 1
                if n_bad <= 2:
                    print("ROUNDTRIP FAIL:", repr(text[:60]))
        self.assertEqual(n_bad, 0, f"{n_bad}/500 benchmark lines failed round-trip")


if __name__ == "__main__":
    unittest.main()


def test_split_planes_mark_only_text_returns_empty():
    """QCRI silver windows can be pure combining marks — must not crash."""
    assert haraqat_planes.split_planes("\u064e\u064b\u064f") == ("", [])
    assert haraqat_planes.split_planes("\u0651\u0652") == ("", [])


def test_split_planes_leading_marks_ride_first_letter():
    skel, labels = haraqat_planes.split_planes("\u0651\u0628")
    assert skel == "\u0628"
    assert labels[0].startswith("\u0651")

def test_split_planes_mark_only_text_returns_empty():
    """QCRI silver windows can be pure combining marks — must not crash."""
    assert haraqat_planes.split_planes("\u064e\u064b\u064f") == ("", [])
    assert haraqat_planes.split_planes("\u0651\u0652") == ("", [])


def test_split_planes_leading_marks_ride_first_letter():
    skel, labels = haraqat_planes.split_planes("\u0651\u0628")
    assert skel == "\u0628"
    assert labels[0].startswith("\u0651")
