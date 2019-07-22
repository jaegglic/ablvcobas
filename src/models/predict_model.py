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
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_scipy,\
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
    b_0_lin, b_1_lin, _, _ = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_scipy, 'rb') as pfile:
    b_0_scipy, b_1_scipy = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_pb, 'rb') as pfile:
    b_0_pb, b_1_pb, _, _ = pickle.load(pfile)

# Generate values for the predictions and save them
X_pred = X_train
y_pred_lin = b_0_lin + b_1_lin * X_pred
y_pred_scipy = b_0_scipy + b_1_scipy * X_pred
y_pred_pb = b_0_pb + b_1_pb * X_pred

# Save predictions
nm_pred_reg_lin  = 'pred_reg_lin.pred'
nm_pred_reg_scipy  = 'pred_reg_scipy.pred'
nm_pred_reg_pb   = 'pred_reg_pb.pred'
with open(PATH_MODELS + nm_pred_reg_lin, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_lin), pfile)
with open(PATH_MODELS + nm_pred_reg_scipy, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_scipy), pfile)
with open(PATH_MODELS + nm_pred_reg_pb, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_pb), pfile)

if __name__ == '__main__':
    # Plot linear regression
    _, ax = plt.subplots(1, 3, figsize=_FIG_SIZE)
    ax[0].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[0].plot(X_pred, y_pred_lin, '--k')
    ax[0].set_title('Linear Reg. (my own)')
    ax[1].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[1].plot(X_pred, y_pred_scipy, '--k')
    ax[1].set_title('Linear Reg. (scipy)')
    ax[2].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[2].plot(X_pred, y_pred_pb, '--k')
    ax[2].set_title('P-B Reg.')

    plt.show()
