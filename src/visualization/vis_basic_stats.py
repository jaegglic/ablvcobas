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
import matplotlib.pyplot as plt
from scipy.stats import variation, kendalltau, spearmanr

# Local imports
import src.utils as utl
from src._paths import PATH_DATA_PROCESSED
from src.features.build_features import nm_data_file_modeling
from src.visualization.plot_specs import set_specs


# Constants
STD_FACT = 1.96
STESABSOLUTILIEBLINGSFARB = (112/255, 28/255, 128/255)

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
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

    str_X = f'{"X ":>{spf}}|' + _str_for_stats(X, spf, prec)
    str_y = f'{"y ":>{spf}}|' + _str_for_stats(y, spf, prec)
    print(str_X)
    print(str_y)

    # Compute correlation statistics
    print('\nCorrelation statistics')
    pear_rho, pear_p, pear_lo, pear_hi = utl.pearsonr_ci(X, y)
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

    fig, ax = plt.subplots(1, 1)
    xlim = [0, 180]
    xstd = np.linspace(xlim[0], xlim[1], 10)

    diff = X - y
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
    ax.text(2, mn - STD_FACT * std - 2, f'-{STD_FACT:.2f}*std',
            verticalalignment='center',
            fontsize=11,
            color=STESABSOLUTILIEBLINGSFARB)
    ax.text(2, mn + STD_FACT * std + 2, f'+{STD_FACT:.2f}*std',
            verticalalignment='center',
            fontsize=11,
            color=STESABSOLUTILIEBLINGSFARB)
    ax.fill_between(xstd, mn + STD_FACT * std, mn - STD_FACT * std,
                    color=STESABSOLUTILIEBLINGSFARB,
                    alpha=0.5
                    )
    plt.show()

