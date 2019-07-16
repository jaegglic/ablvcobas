#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Train the model.
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling

# Constants
_FIG_SIZE = (8, 6)
_SCATTER_SIZE = 1

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X, y = pickle.load(mfile)

# Perform linear regression
X_lin = np.ones((len(X), 2))
X_lin[:, 1] = X
reg_lin = LinearRegression().fit(X_lin, y)

# Perform quadratic regression
X_quad = np.ones((len(X), 3))
X_quad[:, :2] = X_lin
X_quad[:, 2] = X ** 2
reg_quad = LinearRegression().fit(X_quad, y)

# Save trained models
nm_mod_reg_lin = 'mod_reg_lin.mod'
nm_mod_reg_quad = 'mod_reg_quad.mod'
with open(PATH_MODELS + nm_mod_reg_lin, 'wb') as pfile:
    pickle.dump(reg_lin, pfile)
with open(PATH_MODELS + nm_mod_reg_quad, 'wb') as pfile:
    pickle.dump(reg_quad, pfile)


if __name__ == '__main__':
    y_pred_lin = np.dot(X_lin, reg_lin.coef_)
    y_pred_quad = np.dot(X_quad, reg_quad.coef_)

    # Plot linear regression
    _, ax = plt.subplots(1, 2, figsize=_FIG_SIZE)
    ax[0].scatter(X, y, s=_SCATTER_SIZE)
    ax[0].plot(X, y_pred_lin, '--k')
    ax[1].plot((y - y_pred_lin) / y, 'k.')
    print('Linear fit')
    print('  Score  ', reg_lin.score(X_lin, y))
    print('  Coeffs ', reg_lin.coef_)

    # Plot quadratic regression
    _, ax = plt.subplots(1, 2, figsize=_FIG_SIZE)
    ax[0].scatter(X, y, s=_SCATTER_SIZE)
    ax[0].plot(X, y_pred_quad, '--k')
    ax[1].plot((y - y_pred_quad) / y, 'k.')
    print('Quadratic fit')
    print('  Score  ', reg_quad.score(X_quad, y))
    print('  Coeffs ', reg_quad.coef_)

    plt.show()
