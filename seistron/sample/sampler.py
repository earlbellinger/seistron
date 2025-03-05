'''
A script to get the specific columns from the full 311 columns
in the hdf5 files.

Will output the frequency-density plot (4 panels).

02/11/2025: Copied from plot_freq_density.py to now plot HR diagrams.
Needed functionality: to color the data by various quantities, use specific data ranges
02/13/2025: Copied from plot_hr.py to now make echelle diagrams.
02/18/2025: Copied from plot_echelle to now perform linear interpolations.
02/26/2025: Copied from interpolate.py to now sample the space.
02/27/2025: Copied from lin_sampler.py to now sample for knn interpolation.
            Copied from knn_sampler.py to make a general sampler script.
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
import emcee
from fns_for_sampler import *

if(len(sys.argv)<3):
    print("""Usage: python sampler.py 
    1) breakpoint: pre-ms, red-giant 
    2) glob_param: M, Y, Z, alpha, etc. used to slice the data
    3) glob_range: low (<=16th percentile), mean (+/- around mean), high (>=84th percentile), all (all stars)
    4) no. of tracks to select
    5) mode: first, random
    6) nrand: no. of random tracks to use in sampler
    7) interp_type: linear, knn
    """)
    sys.exit(1)

breakpoint = sys.argv[1]
glob_param = sys.argv[2]
glob_range = sys.argv[3]
ntracks = sys.argv[4]
train_split = 0.8 #sys.argv[5]
mode = sys.argv[5].lower()
nrand = sys.argv[6]
interp_type = sys.argv[7]

print(">>> Running slice_h5_files_SUBP.py as subprocess to create hdf5 file. >>>")
subprocess.run(['python', 'slice_h5_files_SUBP.py', breakpoint, glob_param, glob_range, ntracks, mode])

input_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_by_{glob_param}_{glob_range}_range_{mode}.hdf5"

print("\n>>> File and Sample Information >>>")
print("Reading HDF5 file: %s"%input_h5_file)
np.random.seed(42)
df_orig = pd.read_hdf(input_h5_file)
print("File Header:\n",df_orig.head())
df_copy = df_orig.copy()
unique_values, counts = np.unique(df_copy['Track'], return_counts=True)
print("Unique track values, counts in INPUT h5 file:", unique_values, counts)

total_unique_tracks = int(ntracks)+ int(nrand)
print("Total no. of unique tracks (ntracks+nrand):", total_unique_tracks)

if (total_unique_tracks) < len(unique_values):
    tot_selected_tracks = np.random.choice(unique_values, size=int(total_unique_tracks), replace=False)
    nrand_tracks = np.random.choice(tot_selected_tracks, size=int(nrand), replace=False)
    selected_tracks = np.setdiff1d(tot_selected_tracks, nrand_tracks)
else:
    tot_selected_tracks = unique_values
    nrand_tracks = np.random.choice(tot_selected_tracks, size=int(nrand), replace=False)
    selected_tracks = np.setdiff1d(tot_selected_tracks, nrand_tracks)

print("No. of unique track values selected (ntracks):", len(selected_tracks))
print("Unique track no(s).:", selected_tracks)
print("Random track no. used for sampling:", nrand_tracks)

df = df_copy[df_copy['Track'].isin(tot_selected_tracks)]
#print("df (ensuring rows correspond to only selected tracks):", df.shape)

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
print("\n>>> Running curve_fit to obtain delta_nu and nu_max >>>")
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

# making sure that all (ntracks+nrand) have output values calculated from curvefit
df2 = df_copy[df_copy['Track'].isin(tot_selected_tracks)].copy()
df2['epsilon_fit'] = epsilons
df2['delta_nu_fit'] = delta_nus
df2['d0_fit'] = d0s
df2['d1_fit'] = d1s
df2['nu_max'] = nu_max_values

# making a new dataframe with just the ntracks
df3 = df2[df2['Track'].isin(selected_tracks)]
df3 = df3.copy()
df3[df3.isna()] = float('nan')
#print("df3 for ntracks shape:", df3.shape)

less_cols_data = df3[["M", "Y", "Z", "alpha", "fov0_core", "fov0_shell", "Fe_H", "Teff", "luminosity", "delta_nu_fit", "nu_max", "Track"]]
#print("less_cols_data shape:", less_cols_data.head(), less_cols_data.shape)

M_, Y_, Z_ = less_cols_data['M'], less_cols_data['Y'], less_cols_data['Z']
alpha_, fov_core_, fov_shell_ = less_cols_data['alpha'], less_cols_data['fov0_core'], less_cols_data['fov0_shell']
Fe_H_ = less_cols_data['Fe_H']

test_size = 1.0 - float(train_split)
train_data, test_data = train_test_split(less_cols_data, test_size=test_size, random_state=42)

test_pt = np.array([[np.mean(M_), np.mean(Y_), np.mean(Z_), np.mean(alpha_),
                     np.mean(fov_core_), np.mean(fov_shell_)]])

test_points = test_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
true_values = test_data[['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

print("\n>>> Checking grid, preparing for interpolation and MCMC sampling... >>>")
df4 = df2[df2['Track'].isin(nrand_tracks)]
df4 = df4.copy()
df4[df4.isna()] = float('nan')

random_track = df4[["M", "Y", "Z", "alpha", "fov0_core", "fov0_shell", "Fe_H", "Teff", "luminosity", "delta_nu_fit", "nu_max", "Track"]]
#print("random_track\n:", random_track.head(), random_track.shape)

# --------- obtaining data from random track above to use as observation

obs_data = np.asarray([random_track['Fe_H'].iloc[0], random_track['Teff'].iloc[0], random_track['luminosity'].iloc[0], 
                        random_track['delta_nu_fit'].iloc[0], random_track['nu_max'].iloc[0]])
print("\tobs data (Fe_H, Teff, L, delta_nu, nu_max):", obs_data)

obs_errors = np.asarray([0.05, 100, 0.05, 0.5, 0.02]) # Fe/H in [], Teff in K, L in dex, delta nu in uHz, nu max in uHz
print("\tobs errors (Fe_H, Teff, L, delta_nu, nu_max):", obs_errors)

epsilon = 1e-4
param_bounds = [(min(M_)+epsilon, max(M_)-epsilon), (min(Y_)+epsilon, max(Y_)-epsilon),
                (min(Z_)+epsilon, max(Z_)-epsilon), (min(alpha_)+epsilon, max(alpha_)-epsilon),
                (min(fov_core_)+epsilon, max(fov_core_)-epsilon), (min(fov_shell_)+epsilon, max(fov_shell_)-epsilon)]
print("\tinput param bounds (M,Y,Z,alpha,fov_core,fov_shell):", param_bounds)

print("\tChecking grid bounds:", check_grid_bounds(less_cols_data, param_bounds))
print("\tChecking grid coverage:", check_grid_coverage(less_cols_data, param_bounds))

test_p = np.mean(param_bounds, axis=1)  # Pick middle values of bounds

# ----------- interpolation -------------

if interp_type == "linear":
    print("\n>>> Starting linear interpolation... >>>")
    interpolator = linear_interpolator2(train_data, test_pt[0])
    mae = evaluate_interp_accuracy2(train_data, test_points, true_values)
    print("\tMAE:", mae)
    print("\tlog_posterior:", log_posterior(test_p, less_cols_data, obs_data, obs_errors, param_bounds, interp_type=interp_type))
    print("\n>>> Start sampling process... >>>")
    sampler = run_mcmc(less_cols_data, obs_data, obs_errors, param_bounds, interp_type=interp_type)
    samples = sampler.get_chain(discard=1000, thin=10, flat=True)
    print("\tSamples (M,Y,Z,alpha,fov_core,fov_shell):", samples, samples.shape)
if interp_type == "knn":
    print("\n>>> Starting knn interpolation... >>>")
    interpolator = knn_interpolator(train_data, test_pt, n_neighbors=20)
    mae = evaluate_knn_accuracy(train_data, test_points, true_values, n_neighbors=20)
    print("\tMAE (20 neighbors):", mae)
    print("\tlog_posterior:", log_posterior(test_p, less_cols_data, obs_data, obs_errors, param_bounds, interp_type=interp_type))
    print("\n>>> Start sampling process... >>>")
    sampler = run_mcmc(less_cols_data, obs_data, obs_errors, param_bounds, interp_type=interp_type)
    samples = sampler.get_chain(discard=1000, thin=10, flat=True)
    print("\tSamples (M,Y,Z,alpha,fov_core,fov_shell):", samples, samples.shape)


with h5py.File(f"/home/ng474/seistron/results/mcmc_samples_{interp_type}_{breakpoint}_for_{ntracks}_tracks_{nrand}_star.hdf5", "w") as f:
    f.create_dataset("samples", data=samples)

print(f"Successfully saved samples to: /home/ng474/seistron/results/mcmc_samples_{interp_type}_{breakpoint}_for_{ntracks}_tracks_{nrand}_star.hdf5")


