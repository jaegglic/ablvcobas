#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Train the model.
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
from sklearn.linear_model import LinearRegression

# Local imports
import src.utils as utl
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling

# Load data
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'rb') as mfile:
    X, y = pickle.load(mfile)

# Perform linear regression
X_train_lin = np.ones((len(X), 2))
X_train_lin[:, 1] = X
reg_lin = LinearRegression().fit(X_train_lin, y)


# Perform quadratic regression
X_train_quad = np.ones((len(X), 3))
X_train_quad[:, :2] = X_train_lin
X_train_quad[:, 2] = X ** 2
reg_quad = LinearRegression().fit(X_train_quad, y)


# Save trained models
nm_mod_reg_lin  = 'mod_reg_lin.mod'
nm_mod_reg_quad = 'mod_reg_quad.mod'
nm_mod_reg_pb   = 'mod_reg_pb.mod'
with open(PATH_MODELS + nm_mod_reg_lin, 'wb') as pfile:
    pickle.dump(reg_lin, pfile)
with open(PATH_MODELS + nm_mod_reg_quad, 'wb') as pfile:
    pickle.dump(reg_quad, pfile)
with open(PATH_MODELS + nm_mod_reg_pb, 'wb') as pfile:
    b_0, b_1, ci_b_0, ci_b_1 = utl.passing_bablok(X, y)
    pickle.dump((b_0, b_1, ci_b_0, ci_b_1), pfile)

# Create Passing-Bablok Regression Object
reg_pb = LinearRegression(fit_intercept=False)  # This and the next line
reg_pb.intercept_ = 0                           # are important for score
reg_pb.coef_ = np.array([b_0, b_1])


if __name__ == '__main__':

    # Print linear regression
    print('Linear fit')
    print('  Score  ', reg_lin.score(X_train_lin, y))
    print('  Coeffs ', reg_lin.coef_)

    # Print quadratic regression
    print('Quadratic fit')
    print('  Score  ', reg_quad.score(X_train_quad, y))
    print('  Coeffs ', reg_quad.coef_)

    # Print Passing-Bablok regression
    print('Passing-Bablok fit')
    print('  Score  ', reg_pb.score(X_train_lin, y))
    print('  Coeffs ', reg_pb.coef_)
