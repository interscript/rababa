"""Tests for the YallaMorph/CamelMorph aux line builder (r9 teacher).

The aux stream teaches feature-conditional vocalization:
    input  = "MORPH: <ascii feature tokens> | <undiacritized form>"
    target = "<fully diacritized form>"
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_yallamorph_aux import (
    aux_pair,
    build_lines,
    feat_tokens,
    iter_adj_specs,
    iter_noun_specs,
    iter_verb_specs,
    strip_diacritics,
)


class TestStrip(unittest.TestCase):
    def test_strips_haraqat_keeps_base(self):
        self.assertEqual(strip_diacritics("أَلْوَيْتُ"), "ألويت")

    def test_plain_untouched(self):
        self.assertEqual(strip_diacritics("ألت"), "ألت")

    def test_empty_after_strip(self):
        self.assertEqual(strip_diacritics("ًٌٍَُِّْ"), "")


class TestFeatTokens(unittest.TestCase):
    def test_deterministic_order(self):
        a = feat_tokens({"voz": "a", "asp": "pv", "gen": "m"})
        b = feat_tokens({"gen": "m", "asp": "pv", "voz": "a"})
        self.assertEqual(a, b)
        self.assertEqual(a, "asp:pv gen:m voz:a")

    def test_ascii_only(self):
        toks = feat_tokens({"asp": "pv", "per": "1", "num": "s"})
        self.assertTrue(all(ord(c) < 128 for c in toks))


class TestAuxPair(unittest.TestCase):
    FORM = "أَلْوَيْتُ"

    def test_prefix_and_undiacritized_input(self):
        src, tgt = aux_pair({"asp": "pv", "per": "1", "gen": "m", "num": "s", "voz": "a"}, self.FORM)
        self.assertTrue(src.startswith("MORPH: "))
        self.assertIn(" | ", src)
        self.assertTrue(src.endswith("| ألويت"))
        self.assertNotIn("َ", src.split("| ", 1)[1])
        self.assertEqual(tgt, self.FORM)

    def test_rejects_empty_form(self):
        self.assertIsNone(aux_pair({"asp": "pv"}, "ًٌ"))

    def test_rejects_oversize_form(self):
        self.assertIsNone(aux_pair({"asp": "pv"}, "ا" * 300))


class TestSpecs(unittest.TestCase):
    def test_verb_specs_shape(self):
        specs = list(iter_verb_specs())
        self.assertGreater(len(specs), 0)
        keys = {k for s in specs for k in s}
        self.assertTrue(keys <= {"pos", "asp", "per", "gen", "num", "voz", "mod"})
        self.assertTrue(all(s.get("asp") in ("pv", "iv", "c") for s in specs))
        self.assertTrue(all("gen" in s and "num" in s and "per" in s for s in specs))

    def test_noun_specs_shape(self):
        for fn in (iter_noun_specs, iter_adj_specs):
            specs = list(fn())
            self.assertGreater(len(specs), 0)
            keys = {k for s in specs for k in s}
            self.assertTrue(keys <= {"pos", "gen", "num", "cas", "stt"})
            self.assertTrue(all({"gen", "num", "cas", "stt"} <= set(s) for s in specs))


class TestBuildLines(unittest.TestCase):
    def test_dedupe_and_cap(self):
        rows = [({"asp": "pv"}, "فَعَلَ")] * 3 + [({"asp": "iv"}, "يَفْعَلُ")]
        lines = build_lines(rows, cap=10)
        self.assertEqual(len(lines), 2)

    def test_cap_applies(self):
        rows = [({"pos": "v", "asp": "pv", "per": str(i)}, "فَعَلَ") for i in range(100)]
        self.assertEqual(len(build_lines(rows, cap=7)), 7)


if __name__ == "__main__":
    unittest.main()
