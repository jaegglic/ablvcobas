#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file computes the confusion matrix by perturbing the 780 creatinine
levels from the COBAS measurements by a Gaussian error that is deduced from the
variation coefficient.
"""

# Standard library

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Local imports
from src._paths import PATH_DATA_RAW, PATH_FIG
from src.visualization.trial_uncertainty_prop import gfr
from src.visualization.vis_confusion_matrix_kitney import CATEGORIES, \
    print_confusion_matrix, imshow_confusion_matrix
from src.visualization.plot_specs import set_specs

# Constants
TRANSF_CONSTANT = 0.011312217
# VC_SCR = 0.0118    # From Alex's analysis (day-to-day level=92)
VC_SCR = 0.011   # From the specification of the cobas test (at the end of Alex's file)
SEED = 1525
np.random.seed(SEED)


# Load file
nm_data_file_raw = 'GFR_ABL_v_Cobas_1903_1906.xlsx'
df = pd.read_excel(PATH_DATA_RAW + nm_data_file_raw)

if __name__ == '__main__':
    age_set  = np.asarray(df['Alter'])
    ind_mature = age_set >= 18
    age_set = age_set[ind_mature]
    sex_set  = np.asarray(df['Sex'])[ind_mature]
    crea_set = np.asarray(df['Creatinin Cobas (Analyse 53)'])[ind_mature]
    gfr_true = np.asarray(df['Num. eGFR nach CKD-EPI\n(Analyse 64)'])[ind_mature]
    # gfr_true = []

    gfr_test = []
    for age, sex, crea in zip(age_set, sex_set, crea_set):
        crea_std = crea*VC_SCR
        crea_pert = crea + crea_std*np.random.randn()

        # gfr_true.append(gfr(crea, age, sex=sex))
        gfr_test.append(gfr(crea_pert, age, sex=sex))

    gfr_true = np.array(gfr_true)
    gfr_test = np.array(gfr_test)

    # Categorize the values (true and pred)
    gfr_true_cat = np.zeros_like(gfr_true, dtype='int32')
    gfr_test_cat = np.zeros_like(gfr_test, dtype='int32')
    for i_cat, key_cat in enumerate(CATEGORIES):
        bnd = CATEGORIES[key_cat]
        ind_true = np.where(np.logical_and(gfr_true >= bnd[0],
                                           gfr_true < bnd[1]))
        gfr_true_cat[ind_true] = i_cat

        ind_test = np.where(np.logical_and(gfr_test >= bnd[0],
                                           gfr_test < bnd[1]))
        gfr_test_cat[ind_test] = i_cat

    cm = confusion_matrix(gfr_true_cat, gfr_test_cat)
    print_confusion_matrix(cm)
    print(np.sum(np.array(cm), axis=0) / 780)

    classes = np.array(list(CATEGORIES.keys()), dtype='<U10')
    ax = imshow_confusion_matrix(cm, classes)
    set_specs(ax, fig_size=(4, 3))
    fignm = PATH_FIG + 'ConfMat_PerturbedCobas.svg'
    plt.savefig(fignm, format='svg')
    plt.show()

