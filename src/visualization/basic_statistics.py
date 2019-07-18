#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This computes the some basic statistics for each value set. In particular
we compute
    - Mean
    - Variance
    - Coefficient of variation
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
from scipy.stats import variation, pearsonr, kendalltau, spearmanr, norm

# Local imports
from src._paths import PATH_DATA_PROCESSED
from src.features.build_features import nm_data_file_modeling

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X, y = pickle.load(mfile)


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


def _str_for_stats(vals, spf, prec):
    """ Computes the statistics for the given value array. It returns a string
    representation that corresponds to a table with field size `spf`.

    Args:
        vals (ndarray, shape=(nvals,)): The values for which we want to
            compute the statistics
        spf (int): Space per table field.
        prec (int): Precision of floating point numbers.

    Returns:
        str: String representation for the statistics.

    """

    return f'{np.mean(vals):^{spf}.{prec}f}|' \
           f'{np.std(vals):^{spf}.{prec}f}|' \
           f'{variation(vals):^{spf}.{prec}f}|'


if __name__ == '__main__':
    spf = 12       # space per field
    prec = 5       # precision

    head_line = ' '*spf + '|' + \
                f'{"Mean":^{spf}}|' \
                f'{"Std":^{spf}}|' \
                f'{"CV":^{spf}}|'

    print(head_line)
    print('-'*len(head_line))

    str_177 = f'{"177 ":>{spf}}|' + _str_for_stats(X, spf, prec)
    str_64 = f'{"64 ":>{spf}}|' + _str_for_stats(y, spf, prec)
    print(str_177)
    print(str_64)

    # Compute correlation statistics
    print('\nCorrelation statistics')
    pear_rho, pear_p, pear_lo, pear_hi = pearsonr_ci(X, y)
    print(f"  Pearson's:        "
          f"rho = {pear_rho:.{prec}f} "
          f"pval = {pear_p:.{prec}f} "
          f"CI = [{pear_lo:.{prec}f}, {pear_hi:.{prec}f}]")

    print('\nRank correlation statistics')
    kend_t, kend_p = kendalltau(X, y)
    print(f"  Kendal's:         "
          f"tau = {kend_t:.{prec}f} "
          f"pval = {kend_p:.{prec}f}")
    spear_rho, spear_p = spearmanr(X, y)
    print(f"  Spearman's rank:  "
          f"rho = {spear_rho:.{prec}f} "
          f"pval = {spear_p:.{prec}f}")

