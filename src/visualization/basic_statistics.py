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
from scipy.stats import variation, pearsonr, kendalltau

# Local imports
from src._paths import PATH_DATA_PROCESSED
from src.features.build_features import nm_data_file_modeling

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X, y = pickle.load(mfile)


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

    print()
    print(f"Pearson's r:  {pearsonr(X, y)}")
    print(f"Kendal's tau: {kendalltau(X, y)}")

