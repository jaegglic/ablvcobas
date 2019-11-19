#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This file is used to compare the uncertainty obtained from regression
with the empirical uncertainty measured by Alex in "Creatining_Plasma_C1".
"""

# Standard library

# Third party requirements
from scipy.stats import variation

# Local imports

# Constants
TRANSF_CONSTANT = 0.011312217


def gfr(scr, age, scr_unit='mumol/L', sex='female', AA=False):
    """ Estimation of the GFR from the CKD-EPI equation.

    Args:
        scr (float): Serum creatinine level
        age (int): Age of the patient
        scr_unit (str): Unit of the creatinine measure ('mumol/L' or 'mg/dL')
        sex (str): Sex of the patient ('female' or 'male')
        AA (bool): True for afro-american and False for non-afro-american

    Returns:
        float: Estimated GFR value
    """

    # Transform serum creatinine level into the right physical units for the
    # CKD-EPI formula ('mg/dL')
    if scr_unit == 'mumol/L':
        scr = TRANSF_CONSTANT * scr
    elif scr_unit != 'mg/dL':
        raise ValueError(f"scr_unit must either be 'mumol/L' or 'mg/dL'")

    # Check the 'sex' string
    if sex not in ['female', 'F', 'male', 'M']:
        raise ValueError(f"sex must either be 'female', 'F', 'male', or 'M'")

    # Get the formula constants
    C, k, alpha, beta = _gfr_constants(sex=sex, AA=AA)

    return C * (min(scr/k, 1)**alpha) * (max(scr/k, 1)**beta) * 0.993**age


def _gfr_constants(sex='female', AA=False):
    """ Returns formula constants of the GFR estimation by the CDK-EPI
    equation.
    """
    # Check the 'sex' string
    if sex not in ['female', 'F', 'male', 'M']:
        raise ValueError(f"sex must either be 'female', 'F', 'male', or 'M'")

    if sex == 'female' or sex == 'F':
        k = 0.7
        alpha = -0.329
        if AA:
            C = 166
        else:
            C = 144
    else:
        k = 0.9
        alpha = -0.411
        if AA:
            C = 163
        else:
            C = 141

    beta = -1.209

    return C, k, alpha, beta


def _get_vc_scr(scr_level='92', meth='series'):
    """ Uses the values of Alex to compute empirically the varation coefficient
    of serum creatinin measurement.

    Args:
        scr_level (str): Level ('92' or '351')
        meth (str): Measurement method ('series' or 'day2day')

    Returns:
        float: Empirical variation coefficient of serum creatinine
    """
    if scr_level not in ['92', '351']:
        raise ValueError(f'Unknown scr level "{scr_level}"')
    if meth not in ['series', 'day2day']:
        raise ValueError(f'Unknown method "{meth}"')

    if meth == 'series':
        if scr_level == '92':
            measurements = [94, 94, 96, 95, 95, 94, 95, 94, 94, 95]
        else:
            measurements = [366, 364, 368, 365, 367, 367, 367, 367, 367, 370]
    else:
        if scr_level == '92':
            measurements = [94, 94, 95, 94, 96, 98, 96, 96, 96, 96, 95, 95, 96]
        else:
            measurements = [362, 363, 363, 365, 367, 362, 364, 366, 364, 368, 365, 362, 368]

    return variation(measurements)


def _get_vc_gfr(scr, scr_unit='mumol/L', sex='female', AA=False,
                scr_level='92', meth='series'):
    """ Computes the variation coefficient of the estimated GFR value.

    Args:
        scr (float): Serum creatinine level
        age (int): Age of the patient
        scr_unit (str): Unit of the creatinine measure ('mumol/L' or 'mg/dL')
        sex (str): Sex of the patient ('female' or 'male')
        AA (bool): True for afro-american and False for non-afro-american
        scr_level (str): Level ('92' or '351')
        meth (str): Measurement method ('series' or 'day2day')

    Returns:

    """
    vc_scr = _get_vc_scr(scr_level=scr_level, meth=meth)
    # vc_scr = 0.014
    _, k, alpha, beta = _gfr_constants(sex=sex, AA=AA)

    # Transform serum creatinine level into the right physical units for the
    # CKD-EPI formula ('mg/dL')
    if scr_unit == 'mumol/L':
        scr = TRANSF_CONSTANT * scr
    elif scr_unit != 'mg/dL':
        raise ValueError(f"scr_unit must either be 'mumol/L' or 'mg/dL'")

    if scr/k < 1:
        fact = -alpha
    else:
        fact = -beta

    return fact * vc_scr


if __name__ == '__main__':
    test_set = [
        (70, 42, 'male'),
        (82, 64, 'female'),
        (76, 66, 'male'),
        (65, 46, 'female'),
        (168, 66, 'male'),
    ]

    for scr, age, sex in test_set:
        print(f'GFR = {gfr(scr, age, sex=sex):4.0f}')

    print(_get_vc_scr(scr_level='92', meth='series'))
    print(_get_vc_scr(scr_level='351', meth='series'))
    print(_get_vc_scr(scr_level='92', meth='day2day'))
    print(_get_vc_scr(scr_level='351', meth='day2day'))

    print()
    print(_get_vc_gfr(92, 'mumol/L', 'male', False, '92', 'day2day'))



