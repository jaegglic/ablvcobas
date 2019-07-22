#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Train the model.
"""

# Standard library
import pickle

# Third party requirements
from scipy import stats
from sklearn.metrics import r2_score

# Local imports
import src.utils as utl
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X, y = pickle.load(mfile)

# Train models
b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin = utl.lin_reg(X, y)
b_1_scipy, b_0_scipy, _, _, _ = stats.linregress(X, y)
b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb = utl.passing_bablok(X, y)

# Save trained models
nm_mod_reg_lin  = 'mod_reg_lin.mod'
nm_mod_reg_scipy  = 'mod_reg_scipy.mod'
nm_mod_reg_pb   = 'mod_reg_pb.mod'

with open(PATH_MODELS + nm_mod_reg_lin, 'wb') as pfile:
    pickle.dump((b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin), pfile)

with open(PATH_MODELS + nm_mod_reg_scipy, 'wb') as pfile:
    pickle.dump((b_0_scipy, b_1_scipy), pfile)

with open(PATH_MODELS + nm_mod_reg_pb, 'wb') as pfile:
    pickle.dump((b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb), pfile)

if __name__ == '__main__':
    y_pred_lin = b_0_lin + b_1_lin*X
    y_pred_pb = b_0_pb + b_1_pb*X

    # Print linear regression (my own)
    print('Linear fit (my own)')
    print('  Score  ', r2_score(y_pred_lin, y))
    print(f'  b_0 = {b_0_lin} CI = {ci_b_0_lin}')
    print(f'  b_1 = {b_1_lin} CI = {ci_b_1_lin}')

    # Print linear regression (scipy)
    print('Linear fit (scipy)')
    print('  Score  ', r2_score(y_pred_lin, y))
    print(f'  b_0 = {b_0_scipy}')
    print(f'  b_1 = {b_1_scipy}')

    # Print Passing-Bablok regression
    print('Passing-Bablok fit')
    print('  Score  ', r2_score(y_pred_pb, y))
    print(f'  b_0 = {b_0_pb} CI = {ci_b_0_pb}')
    print(f'  b_1 = {b_1_pb} CI = {ci_b_1_pb}')
