'''
A script to get the specific columns from the full 311 columns
in the hdf5 files.

Will output the frequency-density plot (4 panels).

02/11/2025: Copied from plot_freq_density.py to now plot HR diagrams.
Needed functionality: to color the data by various quantities, use specific data ranges
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
import subprocess

if(len(sys.argv)<3):
    print("""Usage: python plot_hr.py 
    1) breakpoint: pre-ms, red-giant 
    2) glob_param: M, Y, Z, alpha, etc. 
    3) glob_range: low (<=16th percentile), mean (+/- around mean), high (>=84th percentile), all (all stars)
    4) no. of tracks 
    5) mode: first, random""")
    sys.exit(1)

breakpoint = sys.argv[1]
glob_param = sys.argv[2]
glob_range = sys.argv[3]
ntracks = sys.argv[4]
mode = sys.argv[5].lower()

print(">>> Running slice_h5_files_SUBP.py as subprocess to create hdf5 file. >>>")
subprocess.run(['python', 'slice_h5_files_SUBP.py', breakpoint, glob_param, glob_range, ntracks, mode])
#print("Successfully created hdf5 file!")

input_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_by_{glob_param}_{glob_range}_range_{mode}.hdf5"
print("Reading HDF5 file: %s"%input_h5_file)

df_orig = pd.read_hdf(input_h5_file)
print("File Header:\n",df_orig.head())

df = df_orig.copy()

nu_columns = sorted([col for col in df.columns if col.startswith('nu_')])
other_columns = [col for col in df.columns if col not in nu_columns]
final_columns = other_columns + nu_columns
df = df[final_columns]

epsilons = []
delta_nus = []
d0s = []
d1s = []

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

        for col in nu_columns:
            if col == 'nu_max':
                continue
            l, n = extract_ln(col)
            predicted_nu = model_func(np.array([[l, n]]), epsilon, delta_nu, d0, d1)
            if np.isnan(row[col]):
                df.at[idx, col] = 0
            else:
                df.at[idx, col] -= predicted_nu[0]

df['epsilon_fit'] = epsilons
df['delta_nu_fit'] = delta_nus
df['d0_fit'] = d0s
df['d1_fit'] = d1s

df2 = df.copy()
df2[df_orig.isna()] = float('nan')

# Need to get unique track no., then color by some variable.

unique_values, counts = np.unique(df2['Track'], return_counts=True)
print("Unique track values:", unique_values)

def add_radius_lines(Rs=[10, 100], Tlower=4000, Tupper=7500, Lpos=20, 
                     sigma=5.67e-5, Lsol=3.839e33, Rsol=6.955e10):
    L_ = lambda R, Teff: 4*np.pi*(R*Rsol)**2*sigma*Teff**4 / Lsol
    T_ = lambda R, L: np.power(L*Lsol/(4*np.pi*(R*Rsol)**2*sigma), 1/4)
    for R in Rs:
        plt.text(np.log10((T_(R, Lpos) * 1.05)), Lpos*0.75, 
                 str(R)+r' R$_\odot$', c='gray', size=16, weight='bold', rotation=-30)
        plt.plot(np.log10([Tlower,        Tupper]), 
                          [L_(R, Tlower), L_(R, Tupper)], 
                 ls='--', c='gray', lw=2, zorder=-999)


np.random.seed(42)  # For reproducibility

filtered_data = df2[df2['Track'].isin(unique_values)]
cmap = plt.get_cmap("magma")
min_M, max_M = filtered_data[glob_param].min(), filtered_data[glob_param].max()
norm = plt.Normalize(vmin=min_M, vmax=max_M)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 

# =========== plotting ===========

plt.figure(figsize=(10, 8))
for track in unique_values:
    data = filtered_data[filtered_data['Track'] == track]
    color = cmap(norm(data[glob_param].iloc[0]))
    plt.plot(data['Teff'], data['luminosity'], color=color, alpha=0.1, label=f'Track {track}')

cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.02)
cbar.set_label(f"{glob_param}", fontsize=14)
plt.xlabel('Effective Temperature (Teff)', fontsize=14)
plt.ylabel('Luminosity', fontsize=14)
plt.gca().invert_xaxis()
plt.tick_params(axis='both',which='both', direction='in', labelsize=12)
#plt.grid(True)
#add_radius_lines(Rs=[10,50,100])
plt.semilogy()

if mode == 'first':
    figure_filesave = get_unique_filename("/home/ng474/seistron/plots/HR_diagram_%s_tracks_by_%s_%s_red-giant"%(ntracks, glob_param, glob_range), ".png")
if mode == 'random':
    figure_filesave = get_unique_filename("/home/ng474/seistron/plots/HR_diagram_%s_tracks_by_%s_%s_red-giant_R"%(ntracks, glob_param, glob_range), ".png")

#figure_filesave = get_unique_filename("/home/ng474/seistron/plots/HR_diagram_red-giant", ".png") #"/home/ng474/seistron/plots/HR_diagram_red-giant.png"
plt.savefig(figure_filesave, dpi=300, bbox_inches='tight')

print(f"Saved figure in: {figure_filesave}")





