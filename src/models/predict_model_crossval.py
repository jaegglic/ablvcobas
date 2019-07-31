#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Predict values for the cross-validation. We use
    - Linear regression
    - Passing-Bablok regression
    - Deming regression
"""


# Standard library
import pickle

# Third party requirements

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.train_model_crossval import NM_KF


if __name__ == '__main__':
    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + NM_KF, 'rb') as kfile:
        kf = pickle.load(kfile)

    # Loop over k-folds
    for k, (_, ind_pred) in enumerate(kf.split(X)):
        X_pred = X[ind_pred]

        # Load trained models
        nm_mod_reg_lin_k = f'mod_reg_lin_k{k:02d}.mod'
        nm_mod_reg_pb_k = f'mod_reg_pb_k{k:02d}.mod'
        nm_mod_reg_dem_k = f'mod_reg_dem_k{k:02d}.mod'
        with open(PATH_MODELS + nm_mod_reg_lin_k, 'rb') as pfile:
            b_0_lin, b_1_lin, _, _ = pickle.load(pfile)
        with open(PATH_MODELS + nm_mod_reg_pb_k, 'rb') as pfile:
            b_0_pb, b_1_pb, _, _ = pickle.load(pfile)
        with open(PATH_MODELS + nm_mod_reg_dem_k, 'rb') as pfile:
            b_0_dem, b_1_dem, _, _, _, _ = pickle.load(pfile)

        # Predict values
        y_pred_lin = b_0_lin + b_1_lin * X_pred
        y_pred_pb = b_0_pb + b_1_pb * X_pred
        y_pred_dem = b_0_dem + b_1_dem * X_pred

        # Save predictions
        nm_pred_reg_lin_k = f'pred_reg_lin_k{k:02d}.pred'
        nm_pred_reg_pb_k = f'pred_reg_pb_k{k:02d}.pred'
        nm_pred_reg_dem_k = f'pred_reg_dem_k{k:02d}.pred'
        with open(PATH_MODELS + nm_pred_reg_lin_k, 'wb') as pfile:
            pickle.dump((X_pred, y_pred_lin), pfile)
        with open(PATH_MODELS + nm_pred_reg_pb_k, 'wb') as pfile:
            pickle.dump((X_pred, y_pred_pb), pfile)
        with open(PATH_MODELS + nm_pred_reg_dem_k, 'wb') as pfile:
            pickle.dump((X_pred, y_pred_dem), pfile)
