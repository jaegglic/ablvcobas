#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file computes confusion matrices based on the GFR groups G1 - G5 given
in:
    https://www.stgag.ch/fachbereiche/kliniken-fuer-innere-medizin/
    klinik-fuer-innere-medizin-am-kantonsspital-frauenfeld/nephrologie/
    informationen-fuer-aerzte-und-zuweiser/
    chronische-nierenerkrankung-definitionstadieneinteilung/

by using a cross-validation strategy.
"""

# Standard library
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS, PATH_FIG
from src.features.build_features import nm_data_file_modeling
from src.visualization.vis_confusion_matrix_kitney import CATEGORIES, \
    print_confusion_matrix, print_agreement, imshow_confusion_matrix
from src.models.train_model_crossval import NM_KF
from src.utils import cohen_kappa, fleiss_kappa, _subj_cat, _cat_rat
from src.visualization.plot_specs import set_specs


# Constants
STD_FACT = 1.96


def print_pred_stats(bias=None, cv=None, rmse=None, mu=None, r2=None):
    """ Print the predictive statistics """
    prec = 8

    if any([bias, cv, rmse, mu, r2]):
        print("Predictive Statistics")
    if bias is not None:
        print(f"    Bias            = {bias:.{prec}f}")
    if cv is not None:
        print(f"    CV              = {cv:.{prec}f}")
    if rmse is not None:
        print(f"    RMSE            = {rmse:.{prec}f}")
    if mu is not None:
        print(f"    MU              = {mu:.{prec}f}")
    if r2 is not None:
        print(f"    R^2             = {r2:.{prec}f}")


if __name__ == '__main__':
    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + NM_KF, 'rb') as kfile:
        kf = pickle.load(kfile)

    # Compute confusion matrix for each split
    n_cat = len(CATEGORIES)
    cm_lin = np.zeros((n_cat, n_cat), dtype='int32')
    cm_pb = np.zeros((n_cat, n_cat), dtype='int32')
    cm_dem = np.zeros((n_cat, n_cat), dtype='int32')
    p_e_lin = 0
    p_e_pb = 0
    p_e_dem = 0

    N, n = len(y), 2
    cat = list(range(n_cat))
    n_ij_lin = np.zeros((N, n_cat))
    n_kj_lin = np.zeros((n_cat, n))
    coh_k_lin = 0
    n_ij_pb = np.zeros((N, n_cat))
    n_kj_pb = np.zeros((n_cat, n))
    coh_k_pb = 0
    n_ij_dem = np.zeros((N, n_cat))
    n_kj_dem = np.zeros((n_cat, n))
    coh_k_dem = 0
    fl_k_lin = 0
    fl_k_pb = 0
    fl_k_dem = 0

    err_lin = 0
    err_sq_lin = 0
    err_pb = 0
    err_sq_pb = 0
    err_dem = 0
    err_sq_dem = 0
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Load predictions
        nm_pred_reg_lin_k  = f'{nm_data_file_modeling}_pred_reg_lin_k{k:02d}.pred'
        nm_pred_reg_pb_k   = f'{nm_data_file_modeling}_pred_reg_pb_k{k:02d}.pred'
        nm_pred_reg_dem_k  = f'{nm_data_file_modeling}_pred_reg_dem_k{k:02d}.pred'
        with open(PATH_MODELS + nm_pred_reg_lin_k, 'rb') as pfile:
            _, y_pred_lin = pickle.load(pfile)
        with open(PATH_MODELS + nm_pred_reg_pb_k, 'rb') as pfile:
            _, y_pred_pb = pickle.load(pfile)
        with open(PATH_MODELS + nm_pred_reg_dem_k, 'rb') as pfile:
            _, y_pred_dem = pickle.load(pfile)

        # Make classes vectors
        y_test_cat = np.zeros_like(y_test)
        y_pred_lin_cat = np.zeros_like(y_pred_lin)
        y_pred_pb_cat = np.zeros_like(y_pred_pb)
        y_pred_dem_cat = np.zeros_like(y_pred_dem)
        for i_cat, key_cat in enumerate(CATEGORIES):
            bnd = CATEGORIES[key_cat]
            ind_test = np.where(np.logical_and(y_test >= bnd[0],
                                               y_test < bnd[1]))
            ind_pred_lin = np.where(np.logical_and(y_pred_lin >= bnd[0],
                                                   y_pred_lin < bnd[1]))
            ind_pred_pb = np.where(np.logical_and(y_pred_pb >= bnd[0],
                                                  y_pred_pb < bnd[1]))
            ind_pred_dem = np.where(np.logical_and(y_pred_dem >= bnd[0],
                                                   y_pred_dem < bnd[1]))

            y_test_cat[ind_test] = i_cat
            y_pred_lin_cat[ind_pred_lin] = i_cat
            y_pred_pb_cat[ind_pred_pb] = i_cat
            y_pred_dem_cat[ind_pred_dem] = i_cat

        # Compute mean of confusion matrix
        labels = range(n_cat)
        cm_lin_i  = confusion_matrix(y_test_cat, y_pred_lin_cat, labels=labels)
        cm_pb_i   = confusion_matrix(y_test_cat, y_pred_pb_cat, labels=labels)
        cm_dem_i  = confusion_matrix(y_test_cat, y_pred_dem_cat, labels=labels)

        cm_lin  += cm_lin_i
        cm_pb   += cm_pb_i
        cm_dem  += cm_dem_i

        # Compute cohen's and fleiss' kappa score
        # The baseline probability is the "agreement by chance" (see p_e in the
        # Wikipedia article https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        n_lin = np.sum(np.sum(cm_lin_i))
        n_pb = np.sum(np.sum(cm_pb_i))
        n_dem = np.sum(np.sum(cm_dem_i))

        rat_lin = np.array([y_test_cat, y_pred_lin_cat]).transpose()
        rat_pb = np.array([y_test_cat, y_pred_pb_cat]).transpose()
        rat_dem = np.array([y_test_cat, y_pred_dem_cat]).transpose()

        coh_k_lin  += cohen_kappa(rat_lin, categories=cat) / kf.n_splits
        n_ij_lin[test_index, :] += _subj_cat(rat_lin, cat)
        n_kj_lin += _cat_rat(rat_lin, cat)
        coh_k_pb   += cohen_kappa(rat_pb, categories=cat) / kf.n_splits
        n_ij_pb[test_index, :] += _subj_cat(rat_pb, cat)
        n_kj_pb += _cat_rat(rat_pb, cat)
        coh_k_dem  += cohen_kappa(rat_dem, categories=cat) / kf.n_splits
        n_ij_dem[test_index, :] += _subj_cat(rat_dem, cat)
        n_kj_dem += _cat_rat(rat_dem, cat)

        fl_k_lin  += fleiss_kappa(rat_lin, categories=cat) / kf.n_splits
        fl_k_pb   += fleiss_kappa(rat_pb, categories=cat) / kf.n_splits
        fl_k_dem  += fleiss_kappa(rat_dem, categories=cat) / kf.n_splits

        # Compute bias, RMSE, and R2
        err_lin += sum(y_pred_lin - y_test)
        err_sq_lin += sum((y_pred_lin - y_test)**2)

        err_pb += sum(y_pred_pb - y_test)
        err_sq_pb += sum((y_pred_pb - y_test)**2)

        err_dem += sum(y_pred_dem - y_test)
        err_sq_dem += sum((y_pred_dem - y_test)**2)

    p_e_lin = sum(np.sum(cm_lin, axis=0) * np.sum(cm_lin, axis=1) / (N**2))
    p_e_pb  = sum(np.sum(cm_pb, axis=0) * np.sum(cm_pb, axis=1) / (N**2))
    p_e_dem = sum(np.sum(cm_dem, axis=0) * np.sum(cm_dem, axis=1) / (N**2))

    bias_lin = err_lin / len(X)
    rmse_lin = np.sqrt(err_sq_lin / len(X))
    r2_lin = 1 - err_sq_lin / sum((y - np.mean(y))**2)

    bias_pb = err_pb / len(X)
    rmse_pb = np.sqrt(err_sq_pb / len(X))
    r2_pb = 1 - err_sq_pb / sum((y - np.mean(y))**2)

    bias_dem = err_dem / len(X)
    rmse_dem = np.sqrt(err_sq_dem / len(X))
    r2_dem = 1 - err_sq_dem / sum((y - np.mean(y))**2)

    p_0_lin_alt = np.mean(np.sum(n_ij_lin*(n_ij_lin-1), axis=1) / n*(n-1) )
    p_e_lin_alt = np.sum(np.prod(n_kj_lin, axis=1)) / (N**n)
    coh_k_lin_alt = (p_0_lin_alt - p_e_lin_alt) / (1 - p_e_lin_alt)

    p_0_pb_alt = np.mean(np.sum(n_ij_pb*(n_ij_pb-1), axis=1) / n*(n-1) )
    p_e_pb_alt = np.sum(np.prod(n_kj_pb, axis=1)) / (N**n)
    coh_k_pb_alt = (p_0_pb_alt - p_e_pb_alt) / (1 - p_e_pb_alt)

    p_0_dem_alt = np.mean(np.sum(n_ij_dem*(n_ij_dem-1), axis=1) / n*(n-1) )
    p_e_dem_alt = np.sum(np.prod(n_kj_dem, axis=1)) / (N**n)
    coh_k_dem_alt = (p_0_dem_alt - p_e_dem_alt) / (1 - p_e_dem_alt)

    # Print confusion matrix
    len_sep = 40
    print('\n\n' + '-' * len_sep)
    print('Linear Regression')
    print('-' * len_sep)
    print_confusion_matrix(cm_lin, title='')
    print('\n')
    print_agreement(p_e_lin, coh_k_lin_alt, fl_k_lin)
    print('\n')
    print_pred_stats(bias=bias_lin, rmse=rmse_lin, r2=r2_lin)
    print('-' * len_sep)

    print('\n\n' + '-' * len_sep)
    print('Passing-Bablok Regression')
    print('-' * len_sep)
    print_confusion_matrix(cm_pb, title='')
    print('\n')
    print_agreement(p_e_pb, coh_k_pb_alt, fl_k_pb)
    print('\n')
    print_pred_stats(bias=bias_pb, rmse=rmse_pb, r2=r2_pb)
    print('-' * len_sep)

    print('\n\n' + '-' * len_sep)
    print('Deming Regression')
    print('-' * len_sep)
    print_confusion_matrix(cm_dem, title='')
    print('\n')
    print_agreement(p_e_dem, coh_k_dem_alt, fl_k_dem)
    print('\n')
    print_pred_stats(bias=bias_dem, rmse=rmse_dem, r2=r2_dem)
    print('-' * len_sep)

    # Plot confusion matrix
    classes = CATEGORIES.keys()
    ax = imshow_confusion_matrix(cm_lin, classes, title='Linear')
    set_specs(ax, fig_size=(4, 3))
    fignm = PATH_FIG + 'ConfMat_CrossVal_Linear.svg'
    plt.savefig(fignm, format='svg')
    ax = imshow_confusion_matrix(cm_pb, classes, title='Passing-Bablok Regression')
    set_specs(ax, fig_size=(4, 3))
    fignm = PATH_FIG + 'ConfMat_CrossVal_Passing.svg'
    plt.savefig(fignm, format='svg')
    ax = imshow_confusion_matrix(cm_dem, classes, title='Deming Regression')
    set_specs(ax, fig_size=(4, 3))
    fignm = PATH_FIG + 'ConfMat_CrossVal_Deming.svg'
    plt.savefig(fignm, format='svg')
    plt.show()


