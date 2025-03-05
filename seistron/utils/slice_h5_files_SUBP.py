
'''
A script to use to slice the h5 files (which were originally the
sub-parquet files based on evolutionary phase).

02/12/2025: Copied fro slice_h5_files.py to now use as a subprocess 
in plotting scripts so I can slice based on Tracks, M, Y, Z, etc.
'''

import pandas as pd
import sys
import h5py
import numpy as np
import subprocess 
import os

if(len(sys.argv)<3):
    print("""Usage: python slice_h5_files_SUBP.py 
    1) breakpoint: pre-ms, red-giant 
    2) glob_param: M, Y, Z, alpha, etc. 
    3) glob_range: low (<=16th percentile), mean (+/- around mean), high (>=84th percentile), all (all stars)
    4) no. of tracks 
    5) mode: first, random""")
    sys.exit(1)

breakpoint = sys.argv[1]
glob_param = sys.argv[2]
glob_range = sys.argv[3]
ntracks = sys.argv[4] #int(sys.argv[4])  
mode = sys.argv[5].lower()

# Input and output file paths
parquet_file = "/home/ng474/seistron/parquets/pre-run-%s.parquet"%breakpoint
print("Original parquet file:", str(parquet_file))


# Input and output file paths
input_h5_file = f"/home/ng474/seistron/hdf5/pre-run-{breakpoint}.hdf5"
output_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_by_{glob_param}_{glob_range}_range_{mode}.hdf5"

if os.path.exists(output_h5_file):
    print(f"File '{output_h5_file}' already exists.")
else:
    print(f"Reading from: {input_h5_file}")

    # Read the HDF5 file
    with pd.HDFStore(input_h5_file, mode='r') as store:
        df = store.get('data')

        if glob_range == "low":
            threshold = df[glob_param].quantile(0.16)
            df = df[df[glob_param] <= threshold]
        elif glob_range == "mean":
            threshold = df[glob_param].quantile(0.50)
            glob_std = df[glob_param].std()
            delta = 0.1 * glob_std
            df = df[(df[glob_param] >= threshold - delta) & (df[glob_param] <= threshold + delta)]
        elif glob_range == "high":
            threshold = df[glob_param].quantile(0.84)
            df = df[df[glob_param] >= threshold]
        elif glob_range == "all":
            df = df

        # Get unique track values
        unique_tracks = df['Track'].unique()
        num_tracks = len(unique_tracks)  # Total unique tracks
        print(f"No. of unique tracks given {glob_param} {glob_range} mask: {num_tracks}")

        # Ensure ntracks does not exceed available tracks
        #ntracks = min(int(ntracks), num_tracks)

        # Select tracks based on mode
        if int(ntracks) >= num_tracks:
            selected_tracks = unique_tracks
            print(f"You've asked for more unique track values than available. The no. of unique tracks will be used.")
        else:
            if mode == "first":
                selected_tracks = unique_tracks[:int(ntracks)]  # Take the first n tracks
                selected_tracks_len = len(selected_tracks)
                print(f"You will use the FIRST {selected_tracks_len} out of {num_tracks} unique tracks available.")
            elif mode == "random":
                selected_tracks = np.random.choice(unique_tracks, size=int(ntracks), replace=False)  # Random n tracks
                selected_tracks_len = len(selected_tracks)
                print(f"You will use a RANDOMLY SELECTED {selected_tracks_len} out of {num_tracks} unique tracks available.")
            else:
                print("Invalid mode. Use 'first' or 'random'.")
                sys.exit(1)

        # Filter rows that belong to the selected tracks
        selected_rows = df[df['Track'].isin(selected_tracks)]

        print(f"Shape of selected rows: {selected_rows.shape}")

    # Save the subset to a new HDF5 file
    selected_rows.to_hdf(output_h5_file, key='data', mode='w', format='table')

    # Verify the new file shape
    with pd.HDFStore(output_h5_file, mode='r') as store:
        print(f"Shape of new HDF5 file: {store.get('data').shape}")

    print(f"All rows from {ntracks} {'randomly selected' if mode == 'random' else 'first'} tracks have been saved to '{output_h5_file}'.")

'''
#with pd.HDFStore(input_h5_file, mode='r') as store:
#    some_rows = store.get('data').iloc[:int(nrows)]
#    print(f"Shape of selected rows: {some_rows.shape}")

#output_h5_file = "/home/ng474/seistron/hdf5/%s_%s_rows.hdf5"%(breakpoint, nrows)
output_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_{mode}_{nrows}_rows.hdf5"

#some_rows.to_hdf(output_h5_file, key='data', mode='w', format='table')

with pd.HDFStore(input_h5_file, mode='r') as store:
    df = store.get('data')
    num_rows = len(df)  # Total number of rows

    # Ensure nrows does not exceed available rows
    nrows = min(nrows, num_rows)

    # Select rows based on mode
    if mode == "first":
        selected_rows = df.iloc[:nrows]
    elif mode == "random":
        selected_rows = df.sample(n=nrows, random_state=42)
    else:
        print("Invalid mode. Use 'first' or 'random'.")
        sys.exit(1)

    print(f"Shape of selected rows: {selected_rows.shape}")


selected_rows.to_hdf(output_h5_file, key='data', mode='w', format='table')

# Verify the new file shape
with pd.HDFStore(output_h5_file, mode='r') as store:
    print(f"Shape of new HDF5 file: {store.get('data').shape}")

print(f"{nrows} {'randomly selected' if mode == 'random' else 'first'} rows have been saved to '{output_h5_file}'.")

'''

