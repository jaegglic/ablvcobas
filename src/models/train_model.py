#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Train the model. We use
    - Linear regression (my own implementation)
    - Linear regression (implementation of scipy)
    - Passing-Bablok regression
    - Deming regression
"""

# Standard library
import pickle

# Third party requirements
from scipy import stats
import numpy as np
from sklearn.metrics import r2_score

# Local imports
import src.utils as utl
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling

# Standard error ratio for deming regression
# STD_RATIO = 1.068797680496178       # Fix point 60vs53
# STD_RATIO = 1.006406118665088       # Fix point 177vs64
STD_RATIO = 1

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
    X, y = pickle.load(mfile)

# Train models
b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin = utl.lin_reg(X, y)
b_1_scipy, b_0_scipy, _, _, _ = stats.linregress(X, y)
b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb = utl.passing_bablok(X, y)
b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem, x_star, y_star = \
    utl.deming_reg(X, y, STD_RATIO**2)

# Save trained models
nm_mod_reg_lin    = f'{nm_data_file_modeling}_mod_reg_lin.mod'
nm_mod_reg_scipy  = f'{nm_data_file_modeling}_mod_reg_scipy.mod'
nm_mod_reg_pb     = f'{nm_data_file_modeling}_mod_reg_pb.mod'
nm_mod_reg_dem    = f'{nm_data_file_modeling}_mod_reg_dem.mod'

with open(PATH_MODELS + nm_mod_reg_lin, 'wb') as pfile:
    pickle.dump((b_0_lin, b_1_lin, ci_b_0_lin, ci_b_1_lin), pfile)

with open(PATH_MODELS + nm_mod_reg_scipy, 'wb') as pfile:
    pickle.dump((b_0_scipy, b_1_scipy), pfile)

with open(PATH_MODELS + nm_mod_reg_pb, 'wb') as pfile:
    pickle.dump((b_0_pb, b_1_pb, ci_b_0_pb, ci_b_1_pb), pfile)

with open(PATH_MODELS + nm_mod_reg_dem, 'wb') as pfile:
    pickle.dump((b_0_dem, b_1_dem, ci_b_0_dem, ci_b_1_dem, x_star, y_star), pfile)

if __name__ == '__main__':
    y_pred_lin = b_0_lin + b_1_lin*X
    y_pred_scipy = b_0_scipy + b_1_scipy*X
    y_pred_pb = b_0_pb + b_1_pb*X

    # Print linear regression (my own)
    print('Linear fit (my own)')
    print('  Score  ', r2_score(y_pred_lin, y))
    print(f'  b_0 = {b_0_lin} CI = {ci_b_0_lin}')
    print(f'  b_1 = {b_1_lin} CI = {ci_b_1_lin}')

    # Print linear regression (scipy)
    print('Linear fit (scipy)')
    print('  Score  ', r2_score(y_pred_scipy, y))
    print(f'  b_0 = {b_0_scipy}')
    print(f'  b_1 = {b_1_scipy}')

    # Print Passing-Bablok regression
    print('Passing-Bablok fit')
    print('  Score  ', r2_score(y_pred_pb, y))
    print(f'  b_0 = {b_0_pb} CI = {ci_b_0_pb}')
    print(f'  b_1 = {b_1_pb} CI = {ci_b_1_pb}')

    # Print Deming regression
    print('Deming fit')
    print('  Score  ', r2_score(y_star, y))
    print(f'  b_0 = {b_0_dem} CI = {ci_b_0_dem}')
    print(f'  b_1 = {b_1_dem} CI = {ci_b_1_dem}')
    err_x = X - (y - b_0_dem) / b_1_dem
    err_y = y - (b_0_dem + b_1_dem*X)
    std_ratio_trained = np.std(err_x) / np.std(err_y)
    print(f'  std ratio = {std_ratio_trained}')
