#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Turn the raw data into features for modeling.
"""

# Standard library
import pickle

# Third party requirements
import pandas as pd

# Local imports
from src._paths import PATH_DATA_RAW, PATH_DATA_PROCESSED

# Load file
nm_data_file_raw = 'GFR_ABL_v_Cobas_1903_1906.xlsx'
df = pd.read_excel(PATH_DATA_RAW + nm_data_file_raw)

# Extract features
X = df['ber. Wert Opus ABL\n(Analyse 177)']
y = df['Num. eGFR nach CKD-EPI\n(Analyse 64)']
if not len(X) == len(y):
    raise ValueError("Number of samples in 'X' and 'y' do not match")

# Save features for modeling
nm_data_file_modeling = 'predict_64_from_177.pdat'
dataset = (X, y)
with open(PATH_DATA_PROCESSED + nm_data_file_modeling, 'wb') as mfile:
    pickle.dump(dataset, mfile)


if __name__ == '__main__':
    print(f'n_samples = {len(X)}')
