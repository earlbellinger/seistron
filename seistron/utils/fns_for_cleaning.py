

'''
Functions to use with cleaning h5 files.
'''

import numpy as np
import pandas as pd

def extract_ln(col):
    parts = col.split('_')
    return int(parts[1]), int(parts[2])

def model_func(x, epsilon, delta_nu, d0, d1):
    l, n = x.T
    nu_npl = (n + l / 2 + epsilon) * delta_nu
    nu_correction = (l * (l + 1) * d0 + d1) * (delta_nu**2) / nu_npl
    return nu_npl - nu_correction

