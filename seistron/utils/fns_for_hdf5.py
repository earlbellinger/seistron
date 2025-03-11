
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


def check_hdf5(breakpoint, nrows):

    """
    To check the structure of HDF5 files (full).

    Parameters:
    -----------
    breakpoint: str
        Either pre-ms or red-giant.
    nrows: int or bool
        Number of rows (int) or None (bool) to use all rows available. 
        The latter option is preferred in this function.

    Returns:
    --------
    None, but ensures that a given file exists and checks its structure.
    """

    if nrows == None:
        #input_h5_file = "/home/ng474/seistron/hdf5/pre-run-%s.hdf5"%breakpoint
        input_h5_file = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/pre-run-%s.hdf5"%breakpoint
        print("Input h5 file:", input_h5_file)
    else:
        #input_h5_file = "/home/ng474/seistron/hdf5/%s_%s_rows.hdf5"%(breakpoint, nrows)
        input_h5_file = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/%s_%s_rows.hdf5"%(breakpoint, nrows)
        print("Input h5 file:", input_h5_file)
    key = 'data'

    try:
        df = pd.read_hdf(input_h5_file, key=key, stop=5)
        print("First five rows (including column names):")
        print(df)
    except KeyError:
        print(f"The key '{key}' was not found in the HDF5 file. Please check the key.")
    except FileNotFoundError:
        print(f"The file '{input_h5_file}' was not found. Please check the file path.")


def check_h5_structure(breakpoint, nrows):

    """
    Checks the structure of an input hdf5 file that is either the 
    full file or split based on the number of rows.

    Input:
        breakpoint (str): either pre-ms or red-giant
        nrows (int): number of rows (e.g. 1000)
    """

    if nrows == None:
        input_h5_file = "../hdf5/pre-run-%s.hdf5"%breakpoint
    else:
        input_h5_file = "../hdf5/%s_%s_rows.hdf5"%(breakpoint, nrows)

    with h5py.File(input_h5_file, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: Dataset, Shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{name}: Group")
                for key in obj.keys():
                    print(f"  {key}: Group/Dataset")


#testing = check_h5_structure("red-giant", 1000)
#testing2 = check_h5_structure("red-giant", None)

#print("testing:", testing)
#print("testing2:", testing2)



def slice_h5(breakpoint, glob_param, glob_range, ntracks, mode):

    """
    To slice the full HDF5 file by various parameters to use in subsequent scripts.

    Parameters:
    -----------
        breakpoint (str): pre-ms, red-giant
        glob_param (str): M, Y, Z, alpha, etc.
        glob_range (str): low (<=16th percentile), mean (+/- around mean), high (>=84th percentile), all (all stars)
        ntracks (int): number of tracks
        mode (str): first, random (either the first ntracks or a random choosing of ntracks)

    Returns:
    -------
    None, but saves new hdf5 file to directory based on function inputs.

    """
    base_dir = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/"
    parquet_file = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/pre-run-%s.parquet"%breakpoint
    print("Original parquet file:", str(parquet_file))

    input_h5_file = base_dir+"data/full_data/pre-run-%s.hdf5"%breakpoint
    output_h5_file = base_dir+f"data/full_data/{breakpoint}_by_{glob_param}_{glob_range}_range_{mode}.hdf5"

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


testing = slice_h5("red-giant", "M", "all", 16000, "random")
print("testing:", testing)


