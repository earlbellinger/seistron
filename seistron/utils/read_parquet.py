

'''
Script to read .parquet files (output of MESA+GYRE data of many models)
and output a subset of the data based on specific cutoffs.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import sys, os
from fns_to_read_parquet import rename_columns
import file_manager

#del(models) # uncomment if there's an issue


# ================ terminal inputs ==================

if(len(sys.argv)<2):
    print("Usage: python read_parquet.py 1) breakpoint: pre-ms, red-giant")

breakpoint = sys.argv[1]

# ===================================================


filename = "/gpfs/gibbs/pi/nagai/ng474/seistron/sim_data/p-mode-freqs.parquet"
print("filename:", filename)

models = pd.read_parquet(filename)

keep = ['star_mass', 'Yinit', 'Zinit', 'amlt', 'fov0_core', 'fov0_shell', 'star_age', 'center_h1',
        'feh', 'luminosity', 'radius', 'Teff', 'log_g', 'eep']
keep += [col for col in models.columns if col.startswith('freq_') or col.startswith('inertia_')]

models = models[keep]

models['Track'], _ = pd.factorize(models['star_mass'])

models.rename(columns={'Yinit': 'Y',
                       'Zinit': 'Z',
                       'amlt': 'alpha',
                       'star_mass': 'M',
                       'feh': 'Fe_H'}, inplace=True)

models.columns = [rename_columns(col) for col in models.columns]
models.columns = [rename_columns(col, pref1='inertia', pref2='E') for col in models.columns]


if breakpoint == "pre-ms":
    models = models[np.logical_and(models['center_h1'] > 0.03, models['Teff'] < 7500, models['luminosity'] < 80)]
if breakpoint == "red-giant":
    models = models[np.logical_and(models['Teff'] < 7500, models['center_h1'] < 0.01)] # changed center_h1 limit from 0.03 to 0.01 NG 09/12/24

new_parquet_filename = "/home/ng474/seistron/parquets/pre-run-%s.parquet"%breakpoint
new_parquet_filename = file_manager.check_and_get_filename(new_parquet_filename)
models.to_parquet(new_parquet_filename, engine='pyarrow')

print("Successfully saved new file to:", new_parquet_filename)

