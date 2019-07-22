#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" TEST FILE
Some tests are run to test the linear regression in `src.utils` function.
"""

# Standard library
import unittest

# Third party requirements
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

# Local imports
import src.utils as utl



# Unit test class
class TestCopyOnWrite(unittest.TestCase):

    def setUp(self) -> None:
        nval = 1000000
        sigma = 0.05

        self.tol = 0.01

        self.b_0 = 20
        self.b_1 = -0.5

        self.x = np.random.rand(nval)
        err = sigma * np.random.randn(nval)
        self.y = self.b_0 + self.b_1 * self.x + err

    def test_lin_reg_vs_scipy(self):
        X = np.ones((len(self.x), 2))
        X[:, 1] = self.x
        sl_scipy, ic_scipy, _, _, _ = stats.linregress(self.x, self.y)
        ic_test, sl_test, ci_b_0, ci_b_1 = utl.lin_reg(self.x, self.y)

        print(f'b_0 = {ic_test} CI = {ci_b_0}')
        print(f'b_1 = {sl_test} CI = {ci_b_1}')

        self.assertTrue(abs(ic_scipy - self.b_0) < self.tol)
        self.assertTrue(abs(sl_scipy - self.b_1) < self.tol)
        self.assertTrue(abs(ic_scipy - ic_test) < self.tol)
        self.assertTrue(abs(sl_scipy - sl_test) < self.tol)

        # _, ax = plt.subplots(1, 1, figsize=(6, 6))
        # ax.scatter(self.x, self.y, s=1)
        # ax.plot(self.x, np.dot(X, np.array([intercept, slope])), 'k.')
        # plt.show()

    def test_lin_reg_vs_sklearn(self):
        X = np.ones((len(self.x), 2))
        X[:, 1] = self.x
        reg_lin_sklearn = LinearRegression(fit_intercept=True).fit(X, self.y)
        ic_sklearn = reg_lin_sklearn.intercept_
        sl_sklearn = reg_lin_sklearn.coef_[1]
        ic_test, sl_test, _, _ = utl.lin_reg(self.x, self.y)


        self.assertTrue(abs(ic_sklearn - self.b_0) < self.tol)
        self.assertTrue(abs(sl_sklearn - self.b_1) < self.tol)
        self.assertTrue(abs(ic_sklearn - ic_test) < self.tol)
        self.assertTrue(abs(sl_sklearn - sl_test) < self.tol)

        # _, ax = plt.subplots(1, 1, figsize=(6, 6))
        # ax.scatter(self.x, self.y, s=1)
        # ax.plot(self.x, np.dot(X, [b_0_sklearn, b_1_sklearn]), 'k.')
        # plt.show()


if __name__ == '__main__':
    unittest.main(verbosity=2)
