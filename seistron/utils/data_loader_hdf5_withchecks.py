import sys
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_loader_hdf5 import *


BASE_DIR = "/home/ng474/seistron/hdf5/"

def test_data_loader(filename, dataset_name, batch_size=32):
    print("\n=== Starting DataLoader Test ===")
    
    # 1. First verify the H5 file exists and is readable
    print(f"\nStep 1: Verifying H5 file...")
    print(f"Testing file: {filename}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"H5 file not found: {filename}")
    
    # 2. Print basic file info
    print(f"\nStep 2: Reading H5 file info...")
    with pd.HDFStore(filename, mode='r') as store:
        print(f"Available keys: {store.keys()}")
        data_shape = store.get(dataset_name.replace('/table', '')).shape
        print(f"Data shape: {data_shape}")
    
    # 3. Create dataset and test basic properties
    print(f"\nStep 3: Creating dataset...")
    dataset = HDF5Dataset(filename, dataset_name)
    print(f"Dataset length: {len(dataset)}")
    
    # 4. Test single item retrieval
    print(f"\nStep 4: Testing single item retrieval...")
    first_item = dataset[0]
    print(f"First item type: {type(first_item)}")
    print(f"First item shape: {first_item.shape if hasattr(first_item, 'shape') else 'N/A'}")
    
    # 5. Create and test DataLoader
    print(f"\nStep 5: Testing DataLoader...")
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          collate_fn=collate_fn)
    print(f"Number of batches: {len(dataloader)}")
    
    # 6. Test batch iteration
    print(f"\nStep 6: Testing batch iteration...")
    for i, batch in enumerate(dataloader):
        if i < 3:  # Only show first 3 batches
            print(f"\nBatch {i}:")
            print(f"Batch type: {type(batch)}")
            print(f"Batch shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}")
            print(f"Batch data range: [{batch.min():.4f}, {batch.max():.4f}]")
        else:
            break
    
    print("\n=== DataLoader Test Completed Successfully ===")
    return True



# for commandline arguments

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Usage: python data_loader_hdf5_withchecks.py 1) breakpoint: pre-ms, red-giant 2) no. of rows")
        sys.exit(1)

    breakpoint = sys.argv[1]
    nrows = sys.argv[2] if len(sys.argv) > 2 else ''

    if nrows == '':
        input_h5_file = f"/home/ng474/seistron/hdf5/pre-run-{breakpoint}.hdf5"
        data_h5_name = f"pre-run-{breakpoint}"
    else:
        input_h5_file = f"/home/ng474/seistron/hdf5/{breakpoint}_{nrows}_rows.hdf5"
        data_h5_name = f"{breakpoint}_{nrows}_rows"

    try:
        success = test_data_loader(input_h5_file, data_h5_name)
        if success:
            print("\nAll tests passed!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        sys.exit(1)


