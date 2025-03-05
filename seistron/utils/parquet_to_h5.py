
'''
Script to transform sub-parquet files (pre-run-...parquet) into hdf5 files
for easier handling.
'''

import pyarrow.parquet as pq
import pandas as pd
import sys
import h5py


if(len(sys.argv)<2):
    print("Usage: python parquet_to_h5.py 1) breakpoint: pre-ms, red-giant")

breakpoint = sys.argv[1]

# Input and output file paths
parquet_file = "/home/ng474/seistron/parquets/pre-run-%s.parquet"%breakpoint
print("Original parquet file:", str(parquet_file))
h5_file = "/home/ng474/seistron/hdf5/pre-run-%s.hdf5"%breakpoint

# Open the Parquet file
parquet_reader = pq.ParquetFile(parquet_file)
print(f"Total row groups in Parquet: {parquet_reader.num_row_groups}")

# First chunk - create the HDF5 file with proper structure
first_chunk = parquet_reader.read_row_group(0).to_pandas()
print(f"First chunk shape: {first_chunk.shape}")
first_chunk.to_hdf(h5_file, key='data', mode='w', format='table')

# Append remaining chunks
for i in range(1, parquet_reader.num_row_groups):
    chunk = parquet_reader.read_row_group(i).to_pandas()
    chunk.to_hdf(h5_file, key='data', mode='a', format='table', append=True)

# Verify final HDF5 file
with pd.HDFStore(h5_file, mode='r') as store:
    final_shape = store.get('data').shape
    print(f"Final HDF5 shape: {final_shape}")
'''
# Iterate through the Parquet file in row group chunks
for i in range(parquet_reader.num_row_groups):
    # Read a chunk (row group) into a DataFrame
    chunk = parquet_reader.read_row_group(i).to_pandas()
    
    # Append the chunk to the HDF5 file
    chunk.to_hdf(h5_file, key='data', mode='a', format='table', append=True)
'''
print("Successfully saved to:", str(h5_file))
