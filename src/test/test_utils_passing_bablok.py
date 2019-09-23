#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TEST FILE
Unit tests for `passing_bablok` in `src.utils` function. The test examples are
taken from

https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/
        NCSS/Passing-Bablok_Regression_for_Method_Comparison.pdf

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
RATIO = 4
ALPHA = 0.05


# Unit test class
class TestCopyOnWrite(unittest.TestCase):
    # Example from https://ncss-wpengine.netdna-ssl.com/wp-content/
    def setUp(self) -> None:
        self.x = [ 7.0,  8.3, 10.5,  9.0,  5.1,  8.2, 10.2, 10.3]
        self.y = [ 7.9,  8.2,  9.6,  9.0,  6.5,  7.3, 10.2, 10.6]

    # Deming Regression Tests
    def test_passing_bablok(self):
        b_0_true, b_1_true = -0.0092, 0.9986
        ci_b_0_true = (-0.7486, 0.7987)
        ci_b_1_true = (0.9856, 1.0099)

        b_0, b_1, ci_b_0, ci_b_1 = \
            utl.passing_bablok(self.y, self.y, alpha=ALPHA)
        print(f'b_0 = {b_0:.4f}, c_i = {ci_b_0}')
        print(f'b_1 = {b_1:.4f}, c_i = {ci_b_1}')

        self.assertAlmostEqual(b_0, b_0_true, places=PREC_DEC)
        self.assertAlmostEqual(b_1, b_1_true, places=PREC_DEC)
        self.assertAlmostEqual(ci_b_0[0], ci_b_0_true[0], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_0[1], ci_b_0_true[1], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_1[0], ci_b_1_true[0], places=PREC_DEC)
        self.assertAlmostEqual(ci_b_1[1], ci_b_1_true[1], places=PREC_DEC)


if __name__ == '__main__':
    unittest.main(verbosity=2)
