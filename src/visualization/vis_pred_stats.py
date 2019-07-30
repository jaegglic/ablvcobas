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
    nm_pred_reg_pb, nm_pred_reg_dem
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_pb, \
    nm_mod_reg_dem

# Constants
STD_FACT = 1.96


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

    bias = np.mean(y_pred - y_test)
    cv = variation(y_pred)
    mu = STD_FACT * np.sqrt(cv**2 + bias**2)
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
    with open(PATH_MODELS + nm_pred_reg_dem, 'rb') as pfile:
        _, y_pred_dem = pickle.load(pfile)

    with open(PATH_MODELS + nm_mod_reg_lin, 'rb') as pfile:
        b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin = pickle.load(pfile)
    with open(PATH_MODELS + nm_mod_reg_pb, 'rb') as pfile:
        b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb = pickle.load(pfile)
    with open(PATH_MODELS + nm_mod_reg_dem, 'rb') as pfile:
        b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem, _, _ = pickle.load(pfile)

    print(head_line)
    print('-'*len(head_line))

    str_lin  = f'{"reg_lin ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_lin, spf, prec)
    str_quad = f'{"reg_scipy ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_scipy, spf, prec)
    str_pb   = f'{"reg_pb ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_pb, spf, prec)
    str_dem   = f'{"reg_dem ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_dem, spf, prec)
    print(str_lin)
    print(str_quad)
    print(str_pb)
    print(str_dem)

    # Linear Regression (my own)
    print('\nLinear Regression')
    print(f'Intercept = {b_0_lin} CI = {ci_b_0_lin}')
    print(f'Slope     = {b_1_lin} CI = {ci_b_1_lin}')

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Linear regression (my own)')
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_lin)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_lin - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X+y)/2, y_pred_lin - y, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - STD_FACT * std, color='red', linestyle='--')
    ax[1].axhline(mn + STD_FACT * std, color='red', linestyle='--')

    # Linear regression (scipy)
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Linear regression (scipy)')
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, y_pred_scipy)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_scipy - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X+y)/2, y_pred_scipy - y, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - STD_FACT * std, color='red', linestyle='--')
    ax[1].axhline(mn + STD_FACT * std, color='red', linestyle='--')

    # Passing-Bablok Regression
    print('\nPassing-Bablok')
    print(f'Intercept = {b_0_pb} CI = {ci_b_0_pb}')
    print(f'Slope     = {b_1_pb} CI = {ci_b_1_pb}')

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Passing-Pablok regression')
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, b_0_pb + X * b_1_pb)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_pb - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X+y)/2, diff, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - STD_FACT * std, color='red', linestyle='--')
    ax[1].axhline(mn + STD_FACT * std, color='red', linestyle='--')

    # Deming Regression
    print('\nDeming Regression')
    print(f'Intercept = {b_0_dem} CI = {ci_b_0_dem}')
    print(f'Slope     = {b_1_dem} CI = {ci_b_1_dem}')

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    fig.suptitle('Passing-Pablok regression')
    # std_b_0_dem = (ci_b_0_dem[1]-ci_b_0_dem[0]) / (2*std_fact)
    # std_b_1_dem = (ci_b_1_dem[1]-ci_b_1_dem[0]) / (2*std_fact)
    # for i in range(1000):
    #     b_0_i = b_0_dem + std_b_0_dem*np.random.randn(1)
    #     b_1_i = b_1_dem + std_b_1_dem*np.random.randn(1)
    #     ax[0].plot(X, b_0_i + b_1_i*X, color=(0.6, 0.6, 0.6))
    ax[0].scatter(X, y, s=1)
    ax[0].plot(X, b_0_pb + X * b_1_pb)
    ax[1].set_title('Bland-Altman Plot')
    diff = y_pred_dem - y
    mn, std = np.mean(diff), np.std(diff)
    ax[1].plot((X + y) / 2, diff, '.k')
    ax[1].axhline(mn, color='red', linestyle='--')
    ax[1].axhline(mn - STD_FACT * std, color='red', linestyle='--')
    ax[1].axhline(mn + STD_FACT * std, color='red', linestyle='--')

    plt.show()
