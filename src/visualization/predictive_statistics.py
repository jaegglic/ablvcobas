#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" In this file we use the prediction schemes to compute statistics between
the true values and the predicted ones. In particular we compute:
    - Mean of error
    - Std of error
    - Bias
    - Root mean square error (equivalent to the measurement uncertainty)
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
from scipy.stats import variation
from sklearn.metrics import mean_squared_error

# Local imports
from src._paths import PATH_DATA_PROCESSED
from src.features.build_features import nm_data_file_modeling

# Load data









def _str_for_pred_stats(y_test, y_pred, spf, prec):
    """ Computes the predictive statistics for the given value arrays. It
    returns a string representation that corresponds to a table with field size
    `spf`.

    Args:
        y_test (ndarray, shape=(nvals,)): True values
        y_pred (ndarray, shape=(nvals,)): Predicted values
        spf (int): Space per table field.
        prec (int): Precision of floating point numbers.

    Returns:
        str: String representation for the statistics.

    """

    return f'{np.mean(y_test - y_pred):^{spf}.{prec}f}|' \
           f'{np.std(y_test - y_pred):^{spf}.{prec}f}|'


if __name__ == '__main__':
    spf = 12       # space per field
    prec = 5       # precision

    head_line = ' '*spf + '|' + \
                f'{"Mean":^{spf}}|' \
                f'{"Std":^{spf}}|'

    print(head_line)
    print('-'*len(head_line))

    str_177 = f'{"177 ":>{spf}}|' + _str_for_stats(X, spf, prec)
    str_64 = f'{"64 ":>{spf}}|' + _str_for_stats(y, spf, prec)
    print(str_177)
    print(str_64)

