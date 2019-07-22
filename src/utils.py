#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This module contains utilities for the analysis of the data as:

    - :meth: `lin_reg`:         Calculates a standard linear regression
    - :meth: `pearsonr_ci`:     Calculates Pearson's r and confidence interval
    - :meth: `passing_bablok`:  Calculates Passing-Bablok regression
    - :meth: `deming_reg`:      Calculates Deming regression

"""

# Standard library
from math import isclose

# Third party requirements
import numpy as np
from scipy.stats import pearsonr, norm

# Local imports


def lin_reg(x, y, alpha=0.05):
    """ This function computes a standard linear regression as proposed in [1].
    In addition it also returns the confidence interval for the two parameters.

    Args:
        x (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        y (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        alpha (float, optional): Significance level

    Returns:
        b_0 (float): Intercept of the linear regression line
        b_1 (float): Slope of the linear regression line
        ci_b_0 (tuple): Confidence interval for b_0
        ci_b_1 (tuple): Confidence interval for b_1

    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    elif n <= 2:
        raise ValueError('x and y must have size > 2.')

    # Estimate linear regression coefficients
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_diff, y_diff = (x - x_mean), (y - y_mean)
    b_1 = sum(x_diff*y_diff) / sum(x_diff**2)
    b_0 = y_mean - b_1*x_mean

    # Estimate uncertainties
    RSS = sum((y - b_0 - b_1*x)**2)
    var_y = RSS / (n-2)
    denom = (n*sum(x**2) - sum(x)**2)
    var_b_0 = var_y * sum(x**2) / denom
    var_b_1 = n*var_y / denom

    # Compute confidence intervals
    z_val = norm.ppf(1 - alpha/2)
    d_b_0 = z_val*np.sqrt(var_b_0)
    d_b_1 = z_val*np.sqrt(var_b_1)
    ci_b_0 = (b_0 - d_b_0, b_0 + d_b_0)
    ci_b_1 = (b_1 - d_b_1, b_1 + d_b_1)

    return b_0, b_1, ci_b_0, ci_b_1


def pearsonr_ci(x, y, alpha=0.05):
    """ This function is taken from
            https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    it calculates the Pearson correlation along with the confidence interval.

    Args:
        x (iterable): Input for correlation calculation
        y (iterable): Input for correlation calculation
        alpha (float, optional): Significance level

    Returns:
        rho (float):  Pearson's correlation coefficient
        pval (float): Probability for non-correlation
        lo (float): Lower bound of confidence interval
        hi (float): Higher bound for confidence interval

    References:
        .. [1] "Pearson correlation coefficient", Wikipedia,
               https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    # Compotute person's r and pval
    rho, pval = pearsonr(x, y)

    # Transform correlation into a Fishers' Z-score (with corresponding std)
    r_z = np.arctanh(rho)
    fishers_std = 1 / np.sqrt(x.size-3)

    # CI under transformation is r_z \pm z_alpha*fishers_std, where z_alpha can
    # be computed from the normal distribution as
    z_alpha = norm.ppf(1 - alpha/2)
    lo_z, hi_z = r_z-z_alpha*fishers_std, r_z+z_alpha*fishers_std

    # Transform the values back using the inverse map
    lo, hi = np.tanh((lo_z, hi_z))

    return rho, pval, lo, hi


def passing_bablok(x, y, alpha=0.05):
    """ Passing-Bablok regression as described in [1]. If

    Args:
        x (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        y (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        alpha (float, optional): Significance level

    Returns:
        b_0 (float): Intercept of the linear regression line
        b_1 (float): Slope of the linear regression line
        ci_b_0 (tuple): Confidence interval for b_0
        ci_b_1 (tuple): Confidence interval for b_1

    References:
        .. [1] H. pssing and W. Bablok. "A new biometrical procedure for
               testing the equality of measurements from two different
               analytical methods". J. Clin Chem. Clin. Biochem., 21:709-720,
               1983.
    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n = len(x)
    N = int(n*(n-1)/2)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    elif n <= 1:
        raise ValueError('x and y must have size > 2.')

    # For each of the n(n-1)/2 pairs of points we compute the slope as
    #       S_ij = (y_j - y_i) / (x_j - x_i)
    S_ij = []
    counter = 0
    for i in range(n):
        for j in range(i+1, n):

            x_i, y_i = x[i], y[i]
            x_j, y_j = x[j], y[j]

            if isclose(x_i, x_j):
                # Distinguish the following cases
                #   - y_j == y_i: omit these pairs (we have 0/0)
                #   - y_j > y_i: put a very large number (we have oo/0)
                #   - y_j < y_i: put a very small number (we have -oo/0)
                if y_j > y_i:
                    S_ij.append(np.inf)
                elif y_j < y_i:
                    S_ij.append(-np.inf)

            else:
                # Distinguish the following cases
                #   - S_ij > -1: add it to the list
                #   - S_ij = -1: omit these pairs
                #   - S_ij < -1: use it and add one to the counter
                S_ij_cand = (y_j-y_i) / (x_j-x_i)
                if S_ij_cand > -1:
                    S_ij.append(S_ij_cand)
                elif S_ij_cand < -1:
                    S_ij.append(S_ij_cand)
                    counter += 1

    # Compute the intercept b_0 and the slope b_1 of the regression line
    S_ij = np.sort(S_ij, kind='mergesort')
    ind_med = np.argwhere(S_ij >= np.median(S_ij))[0]
    b_1 = float(S_ij[ind_med + counter])
    b_0 = np.median(y - b_1*x)

    # Confidence intervals
    z_alpha = norm.ppf(1 - alpha / 2)
    c_alpha = z_alpha * np.sqrt(n*(n-1)*(2*n+5)/18)
    m_lo = int(round((N - c_alpha)/2))
    m_hi = N - m_lo + 1
    ci_b_1 = (S_ij[m_lo + counter], S_ij[m_hi + counter])
    ci_b_0 = (np.median(y - ci_b_1[0]*x), np.median(y - ci_b_1[1]*x))

    return b_0, b_1, ci_b_0, ci_b_1


def deming_reg(x, y, std_ratio, alpha=0.05):
    """ Computes the Deming regression as proposed in
            https://en.wikipedia.org/wiki/Deming_regression.
    
    Args:
        x (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        y (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        std_ratio (float): Ratio of the standard deviation in x and y.
        alpha (float, optional): Significance level

    Returns:
        b_0 (float): Intercept of the linear regression line
        b_1 (float): Slope of the linear regression line
        x_star (ndarray, shape=(n,)): Array for the x_star values
        y_star (ndarray, shape=(n,)): Array for the y_star values

    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    elif n <= 1:
        raise ValueError('x and y must have size > 2.')

    # Solution
    delta = std_ratio ** 2
    b_0, b_1 = _deming_reg(x, y, n, delta)

    # Compute star values
    x_star = x + (b_1/(b_1**2 + delta)) * (y - b_0 - b_1*x)
    y_star = b_0 + b_1*x_star

    # Compute confidence interval
    std_b_0, std_b_1 = _jackknife_deming_reg(x, y, n, delta)
    z_val = norm.ppf(1 - alpha / 2)
    d_b_0 = z_val * std_b_0
    d_b_1 = z_val * std_b_1
    ci_b_0 = (b_0 - d_b_0, b_0 + d_b_0)
    ci_b_1 = (b_1 - d_b_1, b_1 + d_b_1)

    return b_0, b_1, ci_b_0, ci_b_1, x_star, y_star


def _deming_reg(x, y, n, delta):
    """ The very central computation of Deming regression. This is outsourced
    for the jackknife method to be able to call it."""
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_diff, y_diff = (x - x_mean), (y - y_mean)

    s_xx = sum(x_diff ** 2) / (n - 1)
    s_xy = sum(x_diff * y_diff) / (n - 1)
    s_yy = sum(y_diff ** 2) / (n - 1)

    s_yy_ds_xx = s_yy - delta * s_xx
    b_1 = (s_yy_ds_xx + np.sqrt(s_yy_ds_xx ** 2 + 4 * delta * (s_xy ** 2))) / (
                2 * s_xy)
    b_0 = y_mean - b_1 * x_mean
    return b_0, b_1


def _jackknife_deming_reg(x, y, n, delta):
    """ Jackknife method for estimating the standard deviation."""

    b_0, b_1 = _deming_reg(x, y, n, delta)

    b_0_lst, b_1_lst = [], []
    for i in range(n):
        idx = np.concatenate((np.arange(0, i), np.arange(i+1, n)))
        b_0_i, b_1_i = _deming_reg(x[idx], y[idx], n-1, delta)
        b_0_lst.append(b_0_i)
        b_1_lst.append(b_1_i)
    b_0_lst, b_1_lst = np.array(b_0_lst), np.array(b_1_lst)

    b_0_pseudo = n*b_0 - (n-1)*b_0_lst
    b_1_pseudo = n*b_1 - (n-1)*b_1_lst
    b_0_jk = np.mean(b_0_pseudo)
    b_1_jk = np.mean(b_1_pseudo)

    std_b_0 = np.sqrt(np.sum((b_0_pseudo-b_0_jk)**2)/(n*(n-1)))
    std_b_1 = np.sqrt(np.sum((b_1_pseudo-b_1_jk)**2)/(n*(n-1)))

    return std_b_0, std_b_1
