#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Predict values. We use
    - Linear regression (my own implementation)
    - Linear regression (implementation of scipy)
    - Passing-Bablok regression
    - Deming regression
"""

# Standard library
import pickle

# Third party requirements
import matplotlib.pyplot as plt

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.train_model import nm_mod_reg_lin, nm_mod_reg_scipy,\
    nm_mod_reg_pb, nm_mod_reg_dem

# Constants
_N_PRED = 100
_FIG_SIZE = (15, 6)
_SCATTER_SIZE = 1

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
    X_train, y_train = pickle.load(mfile)

# Load trained models
with open(PATH_MODELS + nm_mod_reg_lin, 'rb') as pfile:
    b_0_lin, b_1_lin, _, _ = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_scipy, 'rb') as pfile:
    b_0_scipy, b_1_scipy = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_pb, 'rb') as pfile:
    b_0_pb, b_1_pb, _, _ = pickle.load(pfile)
with open(PATH_MODELS + nm_mod_reg_dem, 'rb') as pfile:
    b_0_dem, b_1_dem, _, _, _, _ = pickle.load(pfile)

# Generate values for the predictions and save them
X_pred = X_train
y_pred_lin = b_0_lin + b_1_lin * X_pred
y_pred_scipy = b_0_scipy + b_1_scipy * X_pred
y_pred_pb = b_0_pb + b_1_pb * X_pred
y_pred_dem = b_0_dem + b_1_dem * X_pred

# Save predictions
nm_pred_reg_lin    = f'{nm_data_file_modeling}_pred_reg_lin.pred'
nm_pred_reg_scipy  = f'{nm_data_file_modeling}_pred_reg_scipy.pred'
nm_pred_reg_pb     = f'{nm_data_file_modeling}_pred_reg_pb.pred'
nm_pred_reg_dem    = f'{nm_data_file_modeling}_pred_reg_dem.pred'
with open(PATH_MODELS + nm_pred_reg_lin, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_lin), pfile)
with open(PATH_MODELS + nm_pred_reg_scipy, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_scipy), pfile)
with open(PATH_MODELS + nm_pred_reg_pb, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_pb), pfile)
with open(PATH_MODELS + nm_pred_reg_dem, 'wb') as pfile:
    pickle.dump((X_pred, y_pred_dem), pfile)

if __name__ == '__main__':
    # Plot linear regression
    _, ax = plt.subplots(1, 4, figsize=_FIG_SIZE)
    ax[0].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[0].plot(X_pred, y_pred_lin, '--k')
    ax[0].set_title('Linear Reg. (my own)')

    ax[1].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[1].plot(X_pred, y_pred_scipy, '--k')
    ax[1].set_title('Linear Reg. (scipy)')

    ax[2].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[2].plot(X_pred, y_pred_pb, '--k')
    ax[2].set_title('P-B Reg.')

    ax[3].scatter(X_train, y_train, s=_SCATTER_SIZE)
    ax[3].plot(X_pred, y_pred_dem, '--k')
    ax[3].set_title('Deming Reg.')

    plt.show()
