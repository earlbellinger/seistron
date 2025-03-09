

'''
Functions to use with cleaning h5 files.
'''

import numpy as np
import pandas as pd

def extract_ln(col):
    """
    In datafiles where column names are e.g. n_0_3, this function
    extracts the information from the numerical parts (0, 3) based
    on the number of _ within the column name. 
    
    For example: This is used for calculating
    delta_nu and nu_max based on which columns have two _, etc.
    """
    parts = col.split('_')
    return int(parts[1]), int(parts[2])

def model_func(x, epsilon, delta_nu, d0, d1):
    """
    Computes the mode frequencies for stellar oscillations using an asymptotic relation
    with a second-order correction.

    Parameters:
    -----------
    x : array-like, shape (N, 2)
        A 2D array where each row contains the angular degree (l) and radial order (n)
        of an oscillation mode.
    epsilon : float
        A parameter accounting for near-surface effects in the frequency scaling.
    delta_nu : float
        The large frequency separation, which is the spacing between consecutive radial orders.
    d0 : float
        A parameter for the second-order correction term dependent on l(l+1).
    d1 : float
        A parameter for the second-order correction independent of l(l+1).

    Returns:
    --------
    numpy.ndarray
        The corrected mode frequencies for the given (l, n) values.

    Notes:
    ------
    The function follows an asymptotic expansion for stellar oscillation frequencies:

        nu_nl = (n + l / 2 + epsilon) * deltanu - [(l (l + 1) d0 + d1) * (deltanu^2) / deltanu_nl]

    The correction term accounts for deviations from the simple scaling relation.
    """
    l, n = x.T
    nu_npl = (n + l / 2 + epsilon) * delta_nu
    nu_correction = (l * (l + 1) * d0 + d1) * (delta_nu**2) / nu_npl
    return nu_npl - nu_correction

