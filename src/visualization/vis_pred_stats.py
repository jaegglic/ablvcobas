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
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS, PATH_FIG
from src.features.build_features import nm_data_file_modeling
from src.models.predict_model import nm_pred_reg_lin, nm_pred_reg_scipy, \
    nm_pred_reg_pb, nm_pred_reg_dem
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_pb, \
    nm_mod_reg_dem
from src.visualization.plot_specs import set_specs

# Constants
STD_FACT = 1.96
STESABSOLUTILIEBLINGSFARB = (112/255, 28/255, 128/255)


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


def _plot_reg_figures(X, y, y_pred, b_0, b_1, ci_b_0, ci_b_1, title):
    # Computations for the uncertainty region
    npts = 100
    x = np.linspace(np.min(X), np.max(X), npts)
    X_mean = np.mean(X)
    y1 = (ci_b_0[0] + b_1*X_mean + ci_b_1[1]*(x - X_mean)) * (x <= X_mean) +\
         (ci_b_0[0] + b_1*X_mean + ci_b_1[0]*(x - X_mean)) * (x > X_mean)
    y2 = (ci_b_0[1] + b_1*X_mean + ci_b_1[0]*(x - X_mean)) * (x <= X_mean) +\
         (ci_b_0[1] + b_1*X_mean + ci_b_1[1]*(x - X_mean)) * (x > X_mean)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)

    ticks_ax0 = [0, 50, 100, 150]

    ax.fill_between(x, y1, y2,
                    color=STESABSOLUTILIEBLINGSFARB,
                    alpha=0.5,
                    )
    ax.scatter(X, y, s=1, color='k')
    ax.plot(x, b_0 + x * b_1, '--r')

    ax.set_aspect('equal', 'box')
    ax.set_xticks(ticks_ax0)
    ax.set_yticks(ticks_ax0)
    set_specs(
        ax,
        x_lim=[0, 180],
        y_lim=[0, 180],
        fig_size=(4, 3),
        aspects=['equal', 'box'])

    fignm = PATH_FIG + title + '_fit.svg'
    plt.savefig(fignm, format='svg')

    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    xlim = [0, 180]
    xstd = np.linspace(xlim[0], xlim[1], 10)

    diff = y_pred - y
    mn, std = np.mean(diff), np.std(diff)
    ax.plot((X + y) / 2, diff, '.k')
    ax.axhline(mn, color='red', linestyle='--')
    ax.axhline(mn - STD_FACT * std,
               color=STESABSOLUTILIEBLINGSFARB,
               linestyle='--')
    ax.axhline(mn + STD_FACT * std,
               color=STESABSOLUTILIEBLINGSFARB,
               linestyle='--')
    set_specs(
        ax,
        fig_size=(2.5, 3),
        x_lim=xlim,
        y_lim=[-20, 20],
        y_ticks=[-20, -10, 0, 10, 20],
    )
    ax.text(2, mn - STD_FACT * std-2, f'-{STD_FACT:.2f}*std',
            verticalalignment='center',
            fontsize=11,
            color=STESABSOLUTILIEBLINGSFARB)
    ax.text(2, mn + STD_FACT * std+2, f'+{STD_FACT:.2f}*std',
            verticalalignment='center',
            fontsize=11,
            color=STESABSOLUTILIEBLINGSFARB)
    ax.fill_between(xstd, mn + STD_FACT * std, mn - STD_FACT * std,
                    color=STESABSOLUTILIEBLINGSFARB,
                    alpha=0.5
                    )

    fignm = PATH_FIG + title + '_BlandAltman.svg'
    plt.savefig(fignm, format='svg')


if __name__ == '__main__':
    spf = 12       # space per field
    prec = 5       # precision

    head_line = ' '*spf + '|' + \
                f'{"Bias":^{spf}}|' \
                f'{"RMSE":^{spf}}|' \
                f'{"MU":^{spf}}|' \
                f'{"R^2":^{spf}}|'

    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + nm_pred_reg_lin, 'rb') as pfile:
        _, y_pred_lin = pickle.load(pfile)
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
    str_pb   = f'{"reg_pb ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_pb, spf, prec)
    str_dem   = f'{"reg_dem ":>{spf}}|' +\
               _str_for_pred_stats(y, y_pred_dem, spf, prec)
    print(str_lin)
    print(str_pb)
    print(str_dem)

    # Linear Regression (my own)
    print('\nLinear Regression')
    print(f'Intercept = {b_0_lin} CI = {ci_b_0_lin}')
    print(f'Slope     = {b_1_lin} CI = {ci_b_1_lin}')
    _plot_reg_figures(X, y, y_pred_lin,
                      b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin,
                      'Linear')

    # Passing-Bablok Regression
    print('\nPassing-Bablok')
    print(f'Intercept = {b_0_pb} CI = {ci_b_0_pb}')
    print(f'Slope     = {b_1_pb} CI = {ci_b_1_pb}')
    _plot_reg_figures(X, y, y_pred_pb,
                      b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb,
                      'Passing')

    # Deming Regression
    print('\nDeming Regression')
    print(f'Intercept = {b_0_dem} CI = {ci_b_0_dem}')
    print(f'Slope     = {b_1_dem} CI = {ci_b_1_dem}')
    _plot_reg_figures(X, y, y_pred_dem,
                      b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem,
                      'Deming')

    plt.show()
