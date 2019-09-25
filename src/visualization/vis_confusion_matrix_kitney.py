#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file computes confusion matrices based on the GFR groups G1 - G5 given
in:
    https://www.stgag.ch/fachbereiche/kliniken-fuer-innere-medizin/
    klinik-fuer-innere-medizin-am-kantonsspital-frauenfeld/nephrologie/
    informationen-fuer-aerzte-und-zuweiser/
    chronische-nierenerkrankung-definitionstadieneinteilung/



ATTENTION:
    !!! It uses the same data for training and testing!!!!!!
"""

# Standard library
from collections import OrderedDict
import pickle

# Third party requirements
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Local imports
from src._paths import PATH_DATA_PROCESSED, PATH_MODELS
from src.features.build_features import nm_data_file_modeling
from src.models.predict_model import nm_pred_reg_lin, nm_pred_reg_scipy, \
    nm_pred_reg_pb, nm_pred_reg_dem
from src.utils import cohen_kappa, fleiss_kappa

# Constants
# CATEGORIES = OrderedDict({
#     'G1':  (90, 999),
#     'G2':  (60, 90),
#     'G3a': (45, 60),
#     'G3b': (30, 45),
#     'G4':  (15, 30),
#     'G5':  ( 0, 15),
# })
CATEGORIES = OrderedDict({
    'G12': (60, 999),
    'G3a': (45, 60),
    'G3b': (30, 45),
    'G4':  (15, 30),
    'G5':  ( 0, 15),
})
# CATEGORIES = OrderedDict({
#     'G12': (60, 999),
#     'G3a': (45, 60),
#     'G3b': (30, 45),
#     'G45': ( 0, 30),
# })
NM_PREDICTORS = [
    nm_pred_reg_lin,
    nm_pred_reg_scipy,
    nm_pred_reg_pb,
    nm_pred_reg_dem,
]


def print_confusion_matrix(cm, normalize=False, title=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Confusion matrix (normalized): {title}")
    else:
        print(f"Confusion matrix (non-normalized): {title}")

    print(cm)
    n_tot = np.sum(np.sum(cm))
    n_true = np.sum(np.diag(cm))
    n_false = n_tot - n_true
    print(f'n_tot            {n_tot:4d}')
    print(f'n_true           {n_true:4d} (p = {n_true / n_tot:.4f})')
    print(f'n_false          {n_false:4d} (p = {n_false / n_tot:.4f})')
    for i in range(1, cm.shape[0]):
        n_false_i_m = np.sum(np.diag(cm, k=-i))
        n_false_i_p = np.sum(np.diag(cm, k=i))
        n_false_i = n_false_i_p + n_false_i_m
        print(f'    dist = {i}     {n_false_i:4d} (p = {n_false_i / n_tot:.4f})')
        if n_false_i > 0:
            print(f'        too low  {n_false_i_p:4d} (p = {n_false_i_p / n_tot:.4f})')
            print(f'        too high {n_false_i_m:4d} (p = {n_false_i_m / n_tot:.4f})')


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title=None,
                          nm_cmap='viridis'):
    """ This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap(nm_cmap)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def print_agreement(base, coh_k, fl_k):
    """ Print the agreement values"""
    prec = 8

    print("Agreement Statistics")
    print(f"    Baseline        = {base:.{prec}f}")
    print(f"    Cohen's kappa   = {coh_k:.{prec}f}")
    print(f"    Fleiss' kappa   = {fl_k:.{prec}f}")


if __name__ == '__main__':
    # Load data
    with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'rb') as mfile:
        _, y_true = pickle.load(mfile)

    y_pred = []
    y_pred_cat = []
    for nm in NM_PREDICTORS:
        with open(PATH_MODELS + nm, 'rb') as pfile:
            _, y_pred_i = pickle.load(pfile)
        y_pred.append(y_pred_i)
        y_pred_cat.append(np.zeros_like(y_pred_i, dtype='int32'))

    # Categorize the values (true and pred)
    y_true_cat = np.zeros_like(y_true, dtype='int32')
    for i_cat, key_cat in enumerate(CATEGORIES):
        bnd = CATEGORIES[key_cat]
        ind_true = np.where(np.logical_and(y_true >= bnd[0],
                                           y_true < bnd[1]))
        y_true_cat[ind_true] = i_cat
        for j, nm_pred in enumerate(NM_PREDICTORS):
            ind_pred = np.where(np.logical_and(y_pred[j] >= bnd[0],
                                               y_pred[j] < bnd[1]))
            y_pred_cat[j][ind_pred] = i_cat

    # Plot confusion matrix
    classes = np.array(list(CATEGORIES.keys()), dtype='<U10')
    categories = np.arange(len(CATEGORIES))
    for y_pred_cat_i, nm_pred in zip(y_pred_cat, NM_PREDICTORS):
        cm = confusion_matrix(y_true_cat, y_pred_cat_i)
        title = nm_pred.split('.')[0]

        print_confusion_matrix(cm, title=title)
        plot_confusion_matrix(y_true_cat, y_pred_cat_i,
                              title=title,
                              classes=classes)

        # Compute and plot cohen's kappa score
        # The baseline probability is the "agreement by chance" (see p_e in the
        # Wikipedia article https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        n = np.sum(np.sum(cm))
        p_e = sum(np.sum(cm, axis=0) * np.sum(cm, axis=1) / (n**2))
        ratings = np.array([y_true_cat, y_pred_cat_i]).transpose()
        coh_k = cohen_kappa(ratings, categories=categories)
        fl_k = fleiss_kappa(ratings, categories=categories)
        print_agreement(p_e, coh_k, fl_k)


    plt.show()



