"""Tests for the RIDE direction-transferability probe math."""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ride_probe_math import cosine_rows, residual_directions


class TestResidualDirections(unittest.TestCase):
    def test_subtraction_per_layer(self):
        base = [[1.0, 2.0], [0.0, 1.0]]
        teach = [[3.0, 5.0], [2.0, 1.0]]
        d = residual_directions(base, teach)
        self.assertEqual(d, [[2.0, 3.0], [2.0, 0.0]])

    def test_layer_count_preserved(self):
        base = [[0.0] * 4 for _ in range(5)]
        teach = [[1.0] * 4 for _ in range(5)]
        self.assertEqual(len(residual_directions(base, teach)), 5)


class TestCosineRows(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(cosine_rows([[1.0, 0.0]], [[2.0, 0.0]])[0], 1.0)

    def test_orthogonal_is_zero(self):
        self.assertAlmostEqual(cosine_rows([[1.0, 0.0]], [[0.0, 3.0]])[0], 0.0)

    def test_opposite_is_minus_one(self):
        self.assertAlmostEqual(cosine_rows([[1.0, 1.0]], [[-2.0, -2.0]])[0], -1.0)

    def test_zero_row_is_zero(self):
        self.assertEqual(cosine_rows([[0.0, 0.0]], [[1.0, 1.0]])[0], 0.0)

    def test_rows_independent(self):
        c = cosine_rows([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]])
        self.assertAlmostEqual(c[0], 1 / math.sqrt(2))
        self.assertAlmostEqual(c[1], 0.0)


if __name__ == "__main__":
    unittest.main()
