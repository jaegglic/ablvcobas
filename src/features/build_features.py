#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Turn the raw data into features for modeling.
"""

# Standard library
import pickle

# Third party requirements
import pandas as pd
import numpy as np

# Local imports
from src._paths import PATH_DATA_RAW, PATH_DATA_PROCESSED

# Choose modeling strategy
nm_data_file_modeling = '60vs53'
# nm_data_file_modeling = '177vs64'


if nm_data_file_modeling == '60vs53':
    var_nm_X = 'Creatinin ABL (Analyse 60)'
    var_nm_y = 'Creatinin Cobas (Analyse 53)'
    f_trans = np.log
    # f_trans = np.asarray
elif nm_data_file_modeling == '177vs64':
    var_nm_X = 'ber. Wert Opus ABL\n(Analyse 177)'
    var_nm_y = 'Num. eGFR nach CKD-EPI\n(Analyse 64)'
    f_trans = np.asarray
else:
    raise ValueError(f'Unknown modeling strategy {nm_data_file_modeling}')

# Load file
nm_data_file_raw = 'GFR_ABL_v_Cobas_1903_1906.xlsx'
df = pd.read_excel(PATH_DATA_RAW + nm_data_file_raw)

# Extract features
X = f_trans(df[var_nm_X])
y = f_trans(df[var_nm_y])


if not len(X) == len(y):
    raise ValueError("Number of samples in 'X' and 'y' do not match")

# Save features for modeling
dataset = (X, y)
with open(PATH_DATA_PROCESSED + nm_data_file_modeling + '.pdat', 'wb') as mfile:
    pickle.dump(dataset, mfile)


if __name__ == '__main__':
    print(f'n_samples = {len(X)}')
