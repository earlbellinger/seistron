'''
A script to read in the hdf5 files created from the parquet files.
Just to make sure that the parquet -> hdf5 transfer went as expected.
'''



import pandas as pd
import sys


if(len(sys.argv)<2):
    print("Usage: python read_hdf5.py 1) breakpoint: pre-ms, red-giant 2) no. of rows")

breakpoint = sys.argv[1]
nrows = sys.argv[2]


if nrows == '':
    input_h5_file = "/home/ng474/seistron/hdf5/pre-run-%s.hdf5"%breakpoint
else:
    input_h5_file = "/home/ng474/seistron/hdf5/%s_%s_rows.hdf5"%(breakpoint, nrows)

# Key to access data in the HDF5 file (adjust if needed)
key = 'data'

try:
    df = pd.read_hdf(input_h5_file, key=key, stop=5)
    print("First five rows (including column names):")
    print(df)
except KeyError:
    print(f"The key '{key}' was not found in the HDF5 file. Please check the key.")
except FileNotFoundError:
    print(f"The file '{input_h5_file}' was not found. Please check the file path.")

