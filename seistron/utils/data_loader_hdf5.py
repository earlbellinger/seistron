
'''
A data loader for hdf5 files (made from parquet files).
'''

import h5py
import numpy as np
import jax.numpy as jnp
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def list_datasets(file_path):
    """
    Recursively lists all dataset names in an HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file.
    
    Returns:
        None: Prints the dataset names.
    """
    def visit_function(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}")

    with h5py.File(file_path, 'r') as hdf_file:
        print(f"Inspecting file: {file_path}")
        hdf_file.visititems(visit_function)


def inspect_dataset(file_path, dataset_name):
    """
    Prints information about a specific dataset in the HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to inspect.
    """
    with h5py.File(file_path, 'r') as hdf_file:
        if dataset_name in hdf_file:
            dataset = hdf_file[dataset_name]
            print(f"Dataset: {dataset_name}")
            print(f"Shape: {dataset.shape}")
            print(f"Data Type: {dataset.dtype}")
            print("Sample Data:", dataset[:5])  # Adjust the slicing as needed
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")

# Custom Dataset for HDF5
class HDF5Dataset(Dataset):
    def __init__(self, file_path, dataset_name):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            dataset_name (str): Name of the dataset inside the HDF5 file.
        """
        self.file_path = file_path
        self.dataset_name = dataset_name

        # Open the file in read-only mode to get the dataset size
        with h5py.File(file_path, 'r') as hdf_file:
            self.dataset_length = len(hdf_file[dataset_name])

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        Returns one sample of data given the index, converted to numpy.
        """
        with h5py.File(self.file_path, 'r') as hdf_file:
            data = hdf_file[self.dataset_name][idx]
        return np.array(data, dtype=np.float32)

# Custom collate function for JAX compatibility
def collate_fn(batch):
    """
    Custom collate function to convert a batch to JAX arrays.
    """
    return jnp.array(batch)


def inspect_dataset(hdf5_file_path, dataset_name):
    """Inspect a dataset in an HDF5 file, handling both regular datasets and pandas tables"""
    with h5py.File(hdf5_file_path, 'r') as f:
        print(f"Dataset: {dataset_name}")
        
        try:
            # Try pandas HDFStore first
            with pd.HDFStore(hdf5_file_path, mode='r') as store:
                data = store.get(dataset_name.replace('/table', ''))
                print(f"Shape: {data.shape}")
                print(f"Data type: pandas DataFrame")
                print("\nFirst few rows:")
                print(data.head())
                
        except (KeyError, AttributeError):
            # If that fails, try regular h5py access
            dataset = f[dataset_name]
            if isinstance(dataset, h5py.Dataset):
                print(f"Shape: {dataset.shape}")
                print(f"Data type: {dataset.dtype}")
                print("\nFirst few elements:")
                print(dataset[:5])  # Print first 5 elements
            else:
                print("This is a group, not a dataset. Available keys:")
                print(list(dataset.keys()))

def verify_h5_structure(file_path):
    """Verify the structure and content of the H5 file"""
    with pd.HDFStore(file_path, mode='r') as store:
        # List all keys in the store
        print(f"Available keys in H5 file: {store.keys()}")
        # Print info about the data
        print("\nDataframe info:")
        print(store.get('data').info())
        # Show first few rows
        print("\nFirst few rows:")
        print(store.get('data').head())
        return store.get('data').shape


