#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Predict values.
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_quad,\
    nm_mod_reg_pb

# Constants
_N_PRED = 100
_FIG_SIZE = (8, 6)
_SCATTER_SIZE = 1

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X_train, y_train = pickle.load(mfile)

# Load trained models
with open(PATH_MODELS + nm_mod_reg_lin, 'rb') as pfile:
    reg_lin = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_quad, 'rb') as pfile:
    reg_quad = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_pb, 'rb') as pfile:
    b_0_reg_pb, b_1_reg_pb, _, _ = pickle.load(pfile)

# Generate values for the predictions and save them
# X_pred = np.linspace(np.min(X_train), np.max(X_train), _N_PRED)
X_pred = X_train
X_pred_lin = np.array([np.ones_like(X_pred), X_pred]).transpose()
y_pred_lin = reg_lin.predict(X_pred_lin)

X_pred_quad = np.array([np.ones_like(X_pred), X_pred, X_pred**2]).transpose()
y_pred_quad = reg_quad.predict(X_pred_quad)

y_pred_pb = b_0_reg_pb + X_train * b_1_reg_pb

# Save predictions
nm_pred_reg_lin  = 'pred_reg_lin.pred'
nm_pred_reg_quad = 'pred_reg_quad.pred'
nm_pred_reg_pb   = 'pred_reg_quad.pb'
with open(PATH_MODELS + nm_pred_reg_lin, 'wb') as pfile:
    pickle.dump((X_pred_lin, y_pred_lin), pfile)
with open(PATH_MODELS + nm_pred_reg_quad, 'wb') as pfile:
    pickle.dump((X_pred_quad, y_pred_quad), pfile)
with open(PATH_MODELS + nm_pred_reg_pb, 'wb') as pfile:
    pickle.dump((X_pred_lin, y_pred_pb), pfile)

if __name__ == '__main__':
    # Plot linear regression
    _, ax = plt.subplots(1, 3, figsize=_FIG_SIZE)
    ax[0].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[0].plot(X_pred, y_pred_lin, '--k')
    ax[0].set_title('Linear Reg.')
    ax[1].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[1].plot(X_pred, y_pred_quad, '--k')
    ax[1].set_title('Quadratic Reg.')
    ax[2].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[2].plot(X_pred, y_pred_pb, '--k')
    ax[2].set_title('P-B Reg.')

    plt.show()
