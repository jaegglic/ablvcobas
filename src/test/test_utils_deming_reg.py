#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TEST FILE
Unit tests for `deming_reg` in `src.utils` function. The test examples are
taken from

https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/
        NCSS/Deming_Regression.pdf

CAUTION:
They might be corrupted because I do not have the appropriate dataset.
"""

# Standard library
import unittest

# Third party requirements
import numpy as np

# Local imports
import src.utils as utl

PREC_DEC = 3
RATIO = 5
ALPHA = 0.05


# Unit test class
class TestCopyOnWrite(unittest.TestCase):
    # Example from https://ncss-wpengine.netdna-ssl.com/wp-content/
    def setUp(self) -> None:
        self.x = [ 7.0,  8.3, 10.5,  9.0,  5.1,  8.2, 10.2, 10.3,  7.1,  5.9]
        self.y = [ 7.9,  8.2,  9.6,  9.0,  6.5,  7.3, 10.2, 10.6,  6.3,  5.2]

        self.x_multi = np.array([
            [34, 35],
            [72, 75],
            [83, 85],
            [102, 104],
            [122, 125],
            [138, 136],
            [152, 152],
            [176, 173],
            [186, 182],
            [215, 212],
        ])
        self.y_multi = np.array([
            [31, 30],
            [50, 46],
            [52, 56],
            [60, 60],
            [84, 84],
            [95, 90],
            [101, 99],
            [115, 116],
            [132, 133],
            [146, 145],
        ])

    # Deming Regression Tests
    def test_deming_reg_known_ratio(self):
        b_0_true, b_1_true = -0.0897449, 1.001194
        ci_b_0_true = (-4.06065, 3.88117)
        ci_b_1_true = (0.56956, 1.43283)

        b_0, b_1, ci_b_0, ci_b_1, x_star, y_star = \
            utl.deming_reg(self.x, self.y, ratio=RATIO, alpha=ALPHA)

        self.assertAlmostEqual(b_0, b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(b_1, b_1_true, places=PREC_DEC)
        self.assertAlmostEqual(ci_b_0[0], ci_b_0_true[0], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_0[1], ci_b_0_true[1], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_1[0], ci_b_1_true[0], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_1[1], ci_b_1_true[1], places=PREC_DEC)

    def test__deming_reg_fattening(self):
        b_0, b_1 = utl._deming_reg(self.x_multi, self.y_multi)
        self.assertTrue(isinstance(b_0, float))
        self.assertTrue(isinstance(b_1, float))

    def test__deming_reg_known_ratio(self):
        b_0_true, b_1_true = -0.0897449, 1.001194

        b_0, b_1 = utl._deming_reg(self.x, self.y, ratio=RATIO)

        self.assertAlmostEqual(b_0, b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(b_1, b_1_true, places=PREC_DEC)

    def test__deming_reg_unknown_ratio(self):
        b_0_true, b_1_true = -1.47179, 0.6855943

        b_0, b_1 = utl._deming_reg(self.x_multi, self.y_multi)

        self.assertAlmostEqual(b_0, b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(b_1, b_1_true, places=PREC_DEC)

    # Jackknife method
    def test__jackknive_fattening(self):
        std_b_0, std_b_1 = utl._jackknife_deming_reg(self.x_multi, self.y_multi)
        self.assertTrue(isinstance(std_b_0, float))
        self.assertTrue(isinstance(std_b_1, float))

    def test__jackknive_known_ratio(self):
        std_b_0_true, std_b_1_true = 1.72199, 0.18718

        std_b_0, std_b_1 = utl._jackknife_deming_reg(self.x, self.y, RATIO)

        self.assertAlmostEqual(std_b_0, std_b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(std_b_1, std_b_1_true, places=PREC_DEC)

    def test__jackknive_unknown_ratio(self):
        std_b_0_true, std_b_1_true = 7.49434, 0.04981

        std_b_0, std_b_1 = utl._jackknife_deming_reg(self.x_multi, self.y_multi)

        self.assertAlmostEqual(std_b_0, std_b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(std_b_1, std_b_1_true, places=PREC_DEC)


if __name__ == '__main__':
    unittest.main(verbosity=2)
