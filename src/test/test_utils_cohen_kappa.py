#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TEST FILE
Some tests are run to test the calculation of `cohen_kappa` in `src.utils`
against the implementation in `sklearn.metrics`.
"""

# Standard library
import unittest

# Third party requirements
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Local imports
import src.utils as utl

PREC_DEC = 8


# Unit test class
class TestCopyOnWrite(unittest.TestCase):

    def setUp(self) -> None:
        n_sampl = 10000
        n_cat = 5
        self.categories = np.array(range(n_cat))
        self.ratings = np.random.randint(n_cat, size=(n_sampl, 2))

    def test_own_against_sklearn(self):
        ratings = self.ratings

        kappa_own = utl.cohen_kappa(ratings, self.categories)
        kappa_sklearn = cohen_kappa_score(ratings[:, 0], ratings[:, 1])

        self.assertAlmostEqual(kappa_own, kappa_sklearn, places=PREC_DEC)


if __name__ == '__main__':
    unittest.main(verbosity=2)
