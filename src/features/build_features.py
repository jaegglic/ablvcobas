#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Turn the raw data into features for modeling.
"""

# Generic packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Linear regression model
from sklearn.linear_model import LinearRegression


colums = ['ber. Wert Opus ABL\n(Analyse 177)',
          'Num. eGFR nach CKD-EPI\n(Analyse 64)']

df = pd.read_excel('..\\..\\data\\raw\\GFR_ABL_v_Cobas_1903_1906.xlsx')


_, ax = plt.subplots(1, 5, figsize=(15, 6))
vals = []
for col in colums:
       vals.append(df[col])

# Perform linear regression
X_lin = np.ones((len(vals[0]), 2))
X_lin[:, 1] = vals[0]
y = np.array(vals[1])
reg_lin = LinearRegression().fit(X_lin, y)

# Perform quadratic regression
X_quad = np.ones((len(vals[0]), 3))
X_quad[:, :2] = X_lin
X_quad[:, 2] = vals[0]**2
reg_quad = LinearRegression().fit(X_quad, y)

print('Linear fit')
print('  Score  ', reg_lin.score(X_lin, y))
print('  Coeffs ', reg_lin.coef_)

y_pred_lin = np.dot(X_lin, reg_lin.coef_)
y_pred_quad = np.dot(X_quad, reg_quad.coef_)

print('Quadratic fit')
print('  Score  ', reg_quad.score(X_quad, y))
print('  Coeffs ', reg_quad.coef_)

ax[0].boxplot(vals)
ax[1].scatter(vals[0], vals[1], s=1)
ax[1].plot(X_lin[:, 1], y_pred_lin, '--k')
ax[2].plot(y - y_pred_lin, 'k.')
ax[3].scatter(vals[0], vals[1], s=1)
ax[3].plot(X_lin[:, 1], y_pred_quad, '--k')
ax[4].plot(y - y_pred_quad, 'k.')


plt.show()
