'''
A script to get the specific columns from the full 311 columns
in the hdf5 files.

Will output the frequency-density plot (4 panels).

02/11/2025: Copied from plot_freq_density.py to now plot HR diagrams.
Needed functionality: to color the data by various quantities, use specific data ranges

02/13/2025: Copied from plot_hr.py to now make echelle diagrams.

02/18/2025: Copied from plot_echelle to now perform linear interpolations.
'''

import sys
import pandas as pd
import h5py
from fns_for_cleaning import *
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from unique_filenames import *
from calc_numax import *
import subprocess
from fns_for_echelle import *
import matplotlib.gridspec as gridspec
from fns_for_lininterp import *
from fns_for_knninterp import *
import cProfile
import pstats

if(len(sys.argv)<3):
    print("""Usage: python interpolate.py 
    1) breakpoint: pre-ms, red-giant 
    2) glob_param: M, Y, Z, alpha, etc. used to slice the data
    3) glob_range: low (<=16th percentile), mean (+/- around mean), high (>=84th percentile), all (all stars)
    4) no. of tracks to select
    5) train_split: percentage of tracks to use for training (e.g. 0.8)
    6) mode: first, random""")
    sys.exit(1)

breakpoint = sys.argv[1]
glob_param = sys.argv[2]
glob_range = sys.argv[3]
ntracks = sys.argv[4]
train_split = sys.argv[5]
mode = sys.argv[6].lower()

print(">>> Running slice_h5_files_SUBP.py as subprocess to create hdf5 file. >>>")
subprocess.run(['python', 'slice_h5_files_SUBP.py', breakpoint, glob_param, glob_range, ntracks, mode])

input_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_by_{glob_param}_{glob_range}_range_{mode}.hdf5"
print("Reading HDF5 file: %s"%input_h5_file)
np.random.seed(42)
df_orig = pd.read_hdf(input_h5_file)
print("File Header:\n",df_orig.head())
df_copy = df_orig.copy()

unique_values, counts = np.unique(df_copy['Track'], return_counts=True)
print("Unique track values, counts in INPUT h5 file:", unique_values, counts)
if int(ntracks) < len(unique_values):
    selected_tracks = np.random.choice(unique_values, size=int(ntracks), replace=False)
else:
    selected_tracks = unique_values

print("No. of unique track values selected (ntracks):", len(selected_tracks))
#print("Unique track no(s).:", selected_tracks)

df = df_copy[df_copy['Track'].isin(selected_tracks)]
print("df (ensuring rows correspond to only selected tracks):", df.shape)

nu_columns = sorted([col for col in df.columns if col.startswith('nu_')])
E_columns = sorted([col for col in df.columns if col.startswith('E_')])
other_columns = [col for col in df.columns if col not in nu_columns]
final_columns = other_columns + nu_columns
df = df[final_columns]

epsilons = []
delta_nus = []
d0s = []
d1s = []
nu_max_values = []

for idx, row in df.iterrows():
    x_data = []
    y_data = []
    for col in nu_columns:
        if col == 'nu_max':
            continue
        l, n = extract_ln(col)
        if not np.isnan(row[col]):
            x_data.append((l, n))
            y_data.append(row[col])

    if len(x_data) > 1:
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        #p0 = [0.5, row['delta_nu_asym'], 0.0001, 0.0001]  # initial guess
        p0 = [0.5, 100, 0.0001, 0.0001]  # initial guess
        #print(row['delta_nu_asym'])
        popt, _ = curve_fit(model_func, x_data, y_data, p0=p0)
        epsilon, delta_nu, d0, d1 = popt

        epsilons.append(epsilon)
        delta_nus.append(delta_nu)
        d0s.append(d0)
        d1s.append(d1)
        nu_max_values.append(nu_max(row['M'], row['radius'], row['Teff']))
        # I don't know why these values are being updated... ask Earl
        for col in nu_columns:
            if col == 'nu_max':
                continue
            l, n = extract_ln(col)
            predicted_nu = model_func(np.array([[l, n]]), epsilon, delta_nu, d0, d1)
            if np.isnan(row[col]):
                df.at[idx, col] = 0
            else:
                df.at[idx, col] -= predicted_nu[0]

#df['epsilon_fit'] = epsilons
#df['delta_nu_fit'] = delta_nus
#df['d0_fit'] = d0s
#df['d1_fit'] = d1s

df2 = df_copy.copy()
df2 = df2[df2['Track'].isin(selected_tracks)]
df2['epsilon_fit'] = epsilons
df2['delta_nu_fit'] = delta_nus
df2['d0_fit'] = d0s
df2['d1_fit'] = d1s
df2['nu_max'] = nu_max_values
print("Obtained delta_nu_fit and nu_max from curve-fit.")
df2[df2.isna()] = float('nan')
#print("df2 n_0_3:", np.asarray(df2['nu_0_3']))


#filtered_data = df2[df2['Track'].isin(selected_tracks)]
#print("filtered_data shape:\n", filtered_data.shape)

less_cols_data = df2[["M", "Y", "Z", "alpha", "fov0_core", "fov0_shell", "Fe_H", "Teff", "luminosity", "delta_nu_fit", "nu_max", "Track"]]
print("less_cols_data shape:\n", less_cols_data.head(), less_cols_data.shape)
#print("nu max filtered data:", filtered_data['nu_max'])

#filename_save = f"/home/ng474/seistron/plots/echelle_model_{timestamp}_track_{track_int}_{mode}.png"

#print("filtered_data[M]:", filtered_data['M'])

# ----------- linear interpolation -------------

M_, Y_, Z_ = less_cols_data['M'], less_cols_data['Y'], less_cols_data['Z']
alpha_, fov_core_, fov_shell_ = less_cols_data['alpha'], less_cols_data['fov0_core'], less_cols_data['fov0_shell']
Fe_H_ = less_cols_data['Fe_H']

test_size = 1.0 - float(train_split)
train_data, test_data = train_test_split(less_cols_data, test_size=test_size, random_state=42)

test_pt = np.array([[np.mean(M_), np.mean(Y_), np.mean(Z_), np.mean(alpha_), 
                     np.mean(fov_core_), np.mean(fov_shell_)]])

lin_int = linear_interpolator2(train_data, test_pt[0])

test_points = test_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
true_values = test_data[['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

mae_linear = evaluate_interp_accuracy2(train_data, test_points, true_values)

print("MAE from linear interpolation:", mae_linear)

# ----------- kNN interpolation -----------

knn_int_5 = knn_interpolator(train_data, test_pt, n_neighbors=5)
knn_int_20 = knn_interpolator(train_data, test_pt, n_neighbors=20)

mae_knn_5 = evaluate_knn_accuracy(train_data, test_points, true_values, n_neighbors=5)
mae_knn_20 = evaluate_knn_accuracy(train_data, test_points, true_values, n_neighbors=20)

print("MAE from kNN interpolation (5 neighbors, weights=distance):", mae_knn_5)
print("MAE from kNN interpolation (20 neighbors, weights=distance):", mae_knn_20)


