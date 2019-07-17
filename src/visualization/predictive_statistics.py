#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" In this file we use the prediction schemes to compute statistics between
the true values and the predicted ones. In particular we compute:
    - Bias
    - Std of error
    - Measurement uncertainty (is this equal to the rmse?)
    - Root mean square error (equivalent to the measurement uncertainty)
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
from sklearn.metrics import mean_squared_error

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.predict_model import nm_pred_reg_lin, nm_pred_reg_quad


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
    k = 2

    bias = np.mean(y_pred - y_test)
    cv = variation(y_pred)
    mu = k * np.sqrt(cv**2 + bias**2)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return f'{bias:^{spf}.{prec}f}|' \
           f'{rmse:^{spf}.{prec}f}|' \
           f'{mu:^{spf}.{prec}f}|'


if __name__ == '__main__':
    spf = 12       # space per field
    prec = 5       # precision

    head_line = ' '*spf + '|' + \
                f'{"Bias":^{spf}}|' \
                f'{"RMSE":^{spf}}|' \
                f'{"MU":^{spf}}|'

    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + nm_pred_reg_lin, 'rb') as pfile:
        X_pred_lin, y_pred_lin = pickle.load(pfile)
    with open(PATH_MODELS + nm_pred_reg_quad, 'rb') as pfile:
        X_pred_quad, y_pred_quad = pickle.load(pfile)

    print(head_line)
    print('-'*len(head_line))

    str_177 = f'{"reg_lin ":>{spf}}|' +\
              _str_for_pred_stats(y, y_pred_lin, spf, prec)
    str_64 = f'{"reg_quad ":>{spf}}|' +\
              _str_for_pred_stats(y, y_pred_quad, spf, prec)
    print(str_177)
    print(str_64)

    _, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_lin)
    ax[1].plot(X, y_pred_lin - y, '.k')

    _, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_quad)
    ax[1].plot(X, y_pred_quad - y, '.k')

    plt.show()

