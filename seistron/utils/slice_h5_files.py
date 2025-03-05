
'''
A script to use to slice the h5 files (which were originally the
sub-parquet files based on evolutionary phase).
'''

import pandas as pd
import sys
import h5py
import numpy as np

if(len(sys.argv)<2):
    print("Usage: python slice_h5_files.py 1) breakpoint: pre-ms, red-giant 2) no. of tracks 3) mode: first, random")
    sys.exit(1)

breakpoint = sys.argv[1]
try:
    ntracks = int(sys.argv[2])  # Convert input to integer
except ValueError:
    print("Error: Number of tracks must be an integer.")
    sys.exit(1)
mode = sys.argv[3].lower()

# Input and output file paths
parquet_file = "/home/ng474/seistron/parquets/pre-run-%s.parquet"%breakpoint
print("Original parquet file:", str(parquet_file))


# Input and output file paths
input_h5_file = f"/home/ng474/seistron/hdf5/pre-run-{breakpoint}.hdf5"
output_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_{mode}_{ntracks}_tracks.hdf5"

print(f"Reading from: {input_h5_file}")

# Read the HDF5 file
with pd.HDFStore(input_h5_file, mode='r') as store:
    df = store.get('data')

    # Get unique track values
    unique_tracks = df['Track'].unique()
    num_tracks = len(unique_tracks)  # Total unique tracks

    # Ensure ntracks does not exceed available tracks
    ntracks = min(ntracks, num_tracks)

    # Select tracks based on mode
    if mode == "first":
        selected_tracks = unique_tracks[:ntracks]  # Take the first n tracks
    elif mode == "random":
        selected_tracks = np.random.choice(unique_tracks, size=ntracks, replace=False)  # Random n tracks
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

