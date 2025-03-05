
'''
To show the entire HDF5 file structure.
'''

import h5py
import sys


if(len(sys.argv)<2):
    print("Usage: python test_data_loader_hdf5.py 1) breakpoint: pre-ms, red-giant 2) no. of rows")

breakpoint = sys.argv[1]
nrows = sys.argv[2]

if nrows == '':
    input_h5_file = "/home/ng474/seistron/hdf5/pre-run-%s.hdf5"%breakpoint
else:
    input_h5_file = "/home/ng474/seistron/hdf5/%s_%s_rows.hdf5"%(breakpoint, nrows)

# Open the file and list all the groups and datasets
with h5py.File(input_h5_file, 'r') as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: Dataset, Shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{name}: Group")
            for key in obj.keys():
                print(f"  {key}: Group/Dataset")
                
    f.visititems(print_structure)

