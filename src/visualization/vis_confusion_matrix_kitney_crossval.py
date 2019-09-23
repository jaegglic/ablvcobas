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
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.visualization.vis_confusion_matrix_kitney import CATEGORIES, \
    print_confusion_matrix
from src.models.train_model_crossval import NM_KF


if __name__ == '__main__':
    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
        X, y = pickle.load(mfile)
    with open(PATH_MODELS + NM_KF, 'rb') as kfile:
        kf = pickle.load(kfile)

    # Compute confusion matrix for each split
    cm_lin = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype='int32')
    cm_pb = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype='int32')
    cm_dem = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype='int32')
    p_e_lin = 0
    p_e_pb = 0
    p_e_dem = 0
    coh_k_lin = 0
    coh_k_pb = 0
    coh_k_dem = 0
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
        labels = range(len(CATEGORIES))
        cm_lin_i  = confusion_matrix(y_test_cat, y_pred_lin_cat, labels=labels)
        cm_pb_i   = confusion_matrix(y_test_cat, y_pred_pb_cat, labels=labels)
        cm_dem_i  = confusion_matrix(y_test_cat, y_pred_dem_cat, labels=labels)

        cm_lin  += cm_lin_i
        cm_pb   += cm_pb_i
        cm_dem  += cm_dem_i

        # Compute and plot cohen's kappa score
        # The baseline probability is the "agreement by chance" (see p_e in the
        # Wikipedia article https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        n_lin = np.sum(np.sum(cm_lin_i))
        n_pb = np.sum(np.sum(cm_pb_i))
        n_dem = np.sum(np.sum(cm_dem_i))

        p_e_lin += sum(np.sum(cm_lin_i, axis=0) * np.sum(cm_lin_i, axis=1) / (n_lin ** 2)) / kf.n_splits
        p_e_pb += sum(np.sum(cm_pb_i, axis=0) * np.sum(cm_pb_i, axis=1) / (n_pb ** 2)) / kf.n_splits
        p_e_dem += sum(np.sum(cm_dem_i, axis=0) * np.sum(cm_dem_i, axis=1) / (n_dem ** 2)) / kf.n_splits

        coh_k_lin += cohen_kappa_score(y_test_cat, y_pred_lin_cat, weights=None) / kf.n_splits
        coh_k_pb += cohen_kappa_score(y_test_cat, y_pred_pb_cat, weights=None) / kf.n_splits
        coh_k_dem += cohen_kappa_score(y_test_cat, y_pred_dem_cat, weights=None) / kf.n_splits

    # Print confusion matrix
    print_confusion_matrix(cm_lin, title='Linear')
    print("\nCohen's kappa")
    print(f"    Baseline        = {p_e_lin:.4f}")
    print(f"    No weights      = {coh_k_lin:.4f}")
    print_confusion_matrix(cm_pb, title='Passing-Bablok')
    print("\nCohen's kappa")
    print(f"    Baseline        = {p_e_pb:.4f}")
    print(f"    No weights      = {coh_k_pb:.4f}")
    print_confusion_matrix(cm_dem, title='Deming')
    print("\nCohen's kappa")
    print(f"    Baseline        = {p_e_dem:.4f}")
    print(f"    No weights      = {coh_k_dem:.4f}")

