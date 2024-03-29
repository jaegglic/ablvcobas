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
    LARGE_NUMBER = np.inf

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
                #   - y_j > y_i: put a very large number (we obtain oo)
                #   - y_j < y_i: put a very small number (we obtain -oo)
                if y_j > y_i:
                    S_ij.append(LARGE_NUMBER)
                elif y_j < y_i:
                    S_ij.append(-LARGE_NUMBER)

            else:
                S_ij_cand = (y_j-y_i) / (x_j-x_i)
                # Distinguish the following cases
                #   - S_ij > -1: add it to the list
                #   - S_ij = -1: omit these pairs
                #   - S_ij < -1: use it and add one to the counter
                if S_ij_cand > -1:
                    S_ij.append(S_ij_cand)
                elif S_ij_cand < -1:
                    S_ij.append(S_ij_cand)
                    counter += 1

    # Compute the intercept b_0 and the slope b_1 of the regression line
    S_ij = np.sort(S_ij, kind='mergesort')
    ind_med = int(np.argwhere(S_ij >= np.median(S_ij))[0])
    b_1 = float(S_ij[ind_med + counter])
    b_0 = np.median(y - b_1*x)

    # Confidence intervals
    z_alpha = norm.ppf(1 - alpha / 2)
    c_alpha = z_alpha * np.sqrt(n*(n-1)*(2*n+5)/18)
    m_lo = int(round((N - c_alpha)/2))
    m_hi = N - m_lo + 1
    ci_b_1 = (S_ij[m_lo + counter], S_ij[m_hi + counter])
    ci_b_0_cand = (np.median(y - ci_b_1[1]*x), np.median(y - ci_b_1[0]*x))
    ci_b_0 = (min(ci_b_0_cand), max(ci_b_0_cand))

    return b_0, b_1, ci_b_0, ci_b_1


def deming_reg(x, y, ratio, alpha=0.05):
    """ Computes the Deming regression as proposed in
            https://en.wikipedia.org/wiki/Deming_regression.

    Args:
        x (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        y (array_like, shape=(n,)): Arrays of values. If the array is not 1-D,
            it will be flattened to 1-D.
        ratio (float): Ratio of the variance in x and y.
        alpha (float, optional): Significance level

    Returns:
        b_0 (float): Intercept of the linear regression line
        b_1 (float): Slope of the linear regression line
        x_star (ndarray, shape=(n,)): Array for the x_star values such that we
            have the "best possible fit" for y_star = b_0 + b_1*x_star
        y_star (ndarray, shape=(n,)): Array for the y_star values

    """

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')
    elif n <= 1:
        raise ValueError('x and y must have size > 2.')

    # Deming regression intercept (b_0) and slope (b_1)
    b_0, b_1 = _deming_reg(x, y, ratio)

    # Compute star values for the best fit in y_star = b_0 + b_1*x_star
    x_star = x + (b_1/(b_1**2 + 1/ratio)) * (y - b_0 - b_1*x)
    y_star = b_0 + b_1*x_star

    # Compute confidence interval
    std_b_0, std_b_1 = _jackknife_deming_reg(x, y, ratio)
    z_val = norm.ppf(1 - alpha / 2)
    d_b_0 = z_val * std_b_0
    d_b_1 = z_val * std_b_1
    ci_b_0 = (b_0 - d_b_0, b_0 + d_b_0)
    ci_b_1 = (b_1 - d_b_1, b_1 + d_b_1)

    return b_0, b_1, ci_b_0, ci_b_1, x_star, y_star


def _deming_reg(x, y, ratio=None):
    """ The very central computation of Deming regression. This is outsourced
    for the jackknife method to be able to call it.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # If ratio is unknown compute an empirical representation of it
    if ratio is None:
        ratio = _variance_ratio(x, y)

    # Flatten the arrays and take all values into account
    x = x.ravel()
    y = y.ravel()

    # Compute the linear regression line
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_diff, y_diff = (x - x_mean), (y - y_mean)

    d_xx = sum(x_diff**2)
    d_yy = sum(y_diff**2)
    d_xy = sum(x_diff * y_diff)

    vardyy_dxx = ratio*d_yy - d_xx
    b_1 = ( vardyy_dxx + np.sqrt(vardyy_dxx**2 + 4*ratio*(d_xy**2)) )\
          / ( 2*ratio*d_xy )
    b_0 = y_mean - b_1*x_mean

    return b_0, b_1


def _variance_ratio(x, y):
    """ Computes the empirical ratio of the variance of x and y (being 2-dim
    arrays).
    """
    if not x.ndim == 2 or not y.ndim == 2:
        raise ValueError('If ratio of variance is unknown, x and y arrays '
                         'must be 2-dimensional')

    n_x, k_x = x.shape
    n_y, k_y = y.shape

    x_mean_i = np.mean(x, axis=1).reshape((n_x, 1))
    y_mean_i = np.mean(y, axis=1).reshape((n_y, 1))

    var_x = sum(np.sum((x - x_mean_i) ** 2, axis=1)) / (n_x * (k_x - 1))
    var_y = sum(np.sum((y - y_mean_i) ** 2, axis=1)) / (n_y * (k_y - 1))

    return var_x / var_y


def _jackknife_deming_reg(x, y, ratio=None):
    """ Jackknife method for estimating the standard deviation from Deming's
    regression.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # If ratio is unknown compute an empirical representation of it
    if ratio is None:
        ratio = _variance_ratio(x, y)

    # Get initial guess
    b_0, b_1 = _deming_reg(x, y, ratio)

    # Flatten the arrays for computing more jackknife guesses
    x = x.ravel()
    y = np.asarray(y).ravel()
    n = x.size

    # Get leave-one-out guesses
    b_0_lst, b_1_lst = [], []
    for i in range(n):
        idx = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        b_0_i, b_1_i = _deming_reg(x[idx], y[idx], ratio)
        b_0_lst.append(b_0_i)
        b_1_lst.append(b_1_i)
    b_0_lst, b_1_lst = np.array(b_0_lst), np.array(b_1_lst)

    # Compute jackknive estimator based on the pseudo-variates
    b_0_pseudo = n*b_0 - (n-1)*b_0_lst
    b_1_pseudo = n*b_1 - (n-1)*b_1_lst
    b_0_jack = np.mean(b_0_pseudo)
    b_1_jack = np.mean(b_1_pseudo)

    # Compute standard-deviation
    std_b_0 = np.sqrt(np.sum((b_0_pseudo-b_0_jack)**2)/(n*(n-1)))
    std_b_1 = np.sqrt(np.sum((b_1_pseudo-b_1_jack)**2)/(n*(n-1)))

    return std_b_0, std_b_1


def _cat_rat(ratings, categories):
    """ n_ki[k,i] represents the number of times rater i predicted category k
    """
    ratings = np.asarray(ratings)
    N, n = ratings.shape
    n_cat = len(categories)

    n_ki = np.zeros((n_cat, n))
    for i_cat, cat in enumerate(categories):
        n_ki[i_cat, :] = np.sum(ratings == cat, axis=0)

    return n_ki


def _subj_cat(ratings, categories):
    """ n_ij[i,j] represents the number of raters who assigned the i-th subject
    to the j-th category
    """
    ratings = np.asarray(ratings)
    N, n = ratings.shape
    n_cat = len(categories)

    n_ij = np.zeros((N, n_cat))
    for i_cat, cat in enumerate(categories):
        n_ij[:, i_cat] = np.sum(ratings == cat, axis=1)

    return n_ij


def cohen_kappa(ratings, categories=None):
    """ Computation of Cohen's kappa for assessing the reliability of agreement
    between a fixed number of raters.

    Args:
        ratings (array_like, shape=(N, n)): N subjects assessed by n raters
        categories (array_like, shape=(n_cat)): Rating categories

    Returns:
        kappa (float): Cohen's kappa value
    """
    ratings = np.asarray(ratings)
    N, n = ratings.shape

    if categories is None:
        categories = np.unique(ratings.ravel())

    n_ij = _subj_cat(ratings, categories)
    p_agree = np.sum(n_ij*(n_ij-1), axis=1) / (n*(n-1))
    p_0 = np.mean(p_agree)

    n_ki = _cat_rat(ratings, categories)
    p_e = np.sum(np.prod(n_ki, axis=1)) / (N**n)

    # Note that 1 - p_e gives the agreement that is attainable above chance,
    # and p_0 - p_e gives the degree of agreement actually achieved above
    # chance
    kappa = (p_0 - p_e) / (1 - p_e)

    return kappa


def fleiss_kappa(ratings, categories=None):
    """ Computation of Fleiss' kappa for assessing the reliability of agreement
    between a fixed number of raters.

    Args:
        ratings (array_like, shape=(N, n)): N subjects assessed by n raters
        categories (array_like, shape=(n_cat)): Rating categories

    Returns:
        kappa (float): Fleiss' kappa value
    """
    ratings = np.asarray(ratings)
    N, n = ratings.shape

    if categories is None:
        categories = np.unique(ratings.ravel())

    n_ij = _subj_cat(ratings, categories)
    p_agree = np.sum(n_ij*(n_ij-1), axis=1) / (n*(n-1))
    p_0 = np.mean(p_agree)

    p_cat = np.sum(n_ij, axis=0) / (N*n)
    p_e = np.sum(p_cat**2)

    # Note that 1 - p_e gives the agreement that is attainable above chance,
    # and p_0 - p_e gives the degree of agreement actually achieved above
    # chance
    kappa = (p_0 - p_e) / (1 - p_e)

    return kappa

