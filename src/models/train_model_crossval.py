#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Train the models for the cross-validation. We use
    - Linear regression
    - Passing-Bablok regression
    - Deming regression
"""

# Standard library
import pickle

# Third party requirements
from sklearn.model_selection import KFold

# Local imports
import src.utils as utl
from src.models.train_model import STD_RATIO
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling

# CONSTANTS
N_SPLITS = 10
NM_KF = 'kfold.kf'


if __name__ == '__main__':
    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
        X, y = pickle.load(mfile)

    # Cross-validation splits
    kf = KFold(n_splits=N_SPLITS, random_state=1, shuffle=False)
    with open(PATH_MODELS + NM_KF, 'wb') as kfile:
        pickle.dump(kf, kfile)
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Compute predictors
        b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin = utl.lin_reg(X_train,
                                                               y_train)
        b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb = utl.passing_bablok(X_train,
                                                                  y_train)
        b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem, x_star, y_star \
            = utl.deming_reg(X_train, y_train, STD_RATIO)

        # Save trained models
        nm_mod_reg_lin_k = f'mod_reg_lin_k{k:02d}.mod'
        nm_mod_reg_pb_k = f'mod_reg_pb_k{k:02d}.mod'
        nm_mod_reg_dem_k = f'mod_reg_dem_k{k:02d}.mod'

        with open(PATH_MODELS + nm_mod_reg_lin_k, 'wb') as pfile:
            pickle.dump((b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin), pfile)
        with open(PATH_MODELS + nm_mod_reg_pb_k, 'wb') as pfile:
            pickle.dump((b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb), pfile)
        with open(PATH_MODELS + nm_mod_reg_dem_k, 'wb') as pfile:
            pickle.dump(
                (b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem, x_star, y_star),
                pfile)
