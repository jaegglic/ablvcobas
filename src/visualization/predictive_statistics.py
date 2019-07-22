#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" In this file we use the prediction schemes to compute statistics between
the true values and the predicted ones. In particular we compute:
    - Bias
    - Root mean square error
    - Measurement uncertainty
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation
from sklearn.metrics import r2_score, mean_squared_error

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.predict_model import nm_pred_reg_lin, nm_pred_reg_scipy, \
    nm_pred_reg_pb
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_pb


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
    r2 = r2_score(y_test, y_pred)

    return f'{bias:^{spf}.{prec}f}|' \
           f'{rmse:^{spf}.{prec}f}|' \
           f'{mu:^{spf}.{prec}f}|' \
           f'{r2:^{spf}.{prec}f}|'


if __name__ == '__main__':
    spf = 12       # space per field
    prec = 5       # precision

    head_line = ' '*spf + '|' + \
                f'{"Bias":^{spf}}|' \
                f'{"RMSE":^{spf}}|' \
                f'{"MU":^{spf}}|' \
                f'{"R^2":^{spf}}|'

    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + nm_pred_reg_lin, 'rb') as pfile:
        _, y_pred_lin = pickle.load(pfile)
    with open(PATH_MODELS + nm_pred_reg_scipy, 'rb') as pfile:
        _, y_pred_scipy = pickle.load(pfile)
    with open(PATH_MODELS + nm_pred_reg_pb, 'rb') as pfile:
        _, y_pred_pb = pickle.load(pfile)

    with open(PATH_MODELS + nm_mod_reg_lin, 'rb') as pfile:
        b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin = pickle.load(pfile)
    with open(PATH_MODELS + nm_mod_reg_pb, 'rb') as pfile:
        b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb = pickle.load(pfile)

    print(head_line)
    print('-'*len(head_line))

    str_lin  = f'{"reg_lin ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_lin, spf, prec)
    str_quad = f'{"reg_scipy ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_scipy, spf, prec)
    str_pb   = f'{"reg_pb ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_pb, spf, prec)
    print(str_lin)
    print(str_quad)
    print(str_pb)


    std_fact = 1.96

    # Linear Regression
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Linear regression (my own)')
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_lin)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_lin - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X+y)/2, y_pred_lin - y, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - std_fact*std, color='red', linestyle='--')
    ax[1].axhline(mn + std_fact*std, color='red', linestyle='--')

    # Quadratic regression
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Linear regression (scipy)')
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_scipy)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_scipy - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X+y)/2, y_pred_scipy - y, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - std_fact*std, color='red', linestyle='--')
    ax[1].axhline(mn + std_fact*std, color='red', linestyle='--')

    # Passing-Bablok Regression
    diff = y_pred_pb - y
    mn, std = np.mean(diff), np.std(diff)

    print('\nLinear Regression')
    print(f'Intercept = {b_0_lin} CI = {ci_b_0_lin}')
    print(f'Slope     = {b_1_lin} CI = {ci_b_1_lin}')

    print('\nPassing-Bablok')
    print(f'Intercept = {b_0_pb} CI = {ci_b_0_pb}')
    print(f'Slope     = {b_1_pb} CI = {ci_b_1_pb}')

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Passing-Pablok regression')
    # for _ in range(1000):
    #     b_0i = ci_b_0[0] + (ci_b_0[1] - ci_b_0[0])*np.random.rand()
    #     b_1i = ci_b_1[0] + (ci_b_1[1] - ci_b_1[0])*np.random.rand()
    #     ax[0].plot(X, b_0i + X*b_1i, color=(0.6, 0.6, 0.6))
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, b_0_pb + X * b_1_pb)
    ax[1].set_title('Bland-Altman Plot')
    ax[1].plot((X+y)/2, diff, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - std_fact*std, color='red', linestyle='--')
    ax[1].axhline(mn + std_fact*std, color='red', linestyle='--')

    plt.show()
