import sys
import pandas as pd
import os

"""
A general script to read SLICED .parquet files based on stellar evolution
conditions for ms, red-giant, etc.

As long as the parquet file exists, then this should work (and it does)!

DO NOT MODIFY THIS SCRIPT, MAKE A COPY FIRST!
-------------------------------------------------------------------------
Naomi Gluck | Yale University 2024
"""

base_dir = "/home/ng474/seistron/parquets/"

def list_parquet_files(directory):
    """List all Parquet files in the specified directory."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        if files:
            print("Parquet files in the directory:")
            for file in files:
                print(file)
        else:
            print("No Parquet files found in the directory.")
    except Exception as e:
        print(f"Error accessing the directory: {directory}")
        print(e)

if len(sys.argv) < 2:
    print("Usage: python load_and_save_column.py 1) filename (without base_dir or .parquet)")
    print("Files in /parquet directory are:", list_parquet_files(base_dir))

parquet_file_path = sys.argv[1] + ".parquet"
print(".parquet file path:", base_dir + parquet_file_path)


models = pd.read_parquet(base_dir + parquet_file_path, engine='pyarrow')

asdf = models[models.Track == 53]
print("testing models.Track:", asdf)


