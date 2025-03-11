import re


def rename_columns(col, pref1='freq', pref2='nu'):
    """
    Renames a column by replacing a specific prefix pattern with a new one.

    Parameters:
    -----------
    col : str
        The column name to be processed.
    pref1 : str, optional
        The prefix to search for in the column name (default is 'freq').
    pref2 : str, optional
        The prefix to replace `pref1` with (default is 'nu').

    Returns:
    --------
    str
        The modified column name if it matches the expected pattern; otherwise, returns the original name.

    """
    match = re.match(pref1 + r"_l(\d+)_n(\d+)", col)
    if match:
        return pref2 + f"_{match.group(1)}_{match.group(2)}"
    else:
        return col

def extract_numbers(col):
    """
    Extracts numerical indices from column names following the pattern 'nu_<l>_<n>'.

    Parameters:
    -----------
    col : str
        The column name to be processed.

    Returns:
    --------
    tuple of (int, int)
        A tuple `(l, n)` where `l` and `n` are integers extracted from the column name.
        Returns `(float('inf'), float('inf'))` if no match is found.
    """
    matches = re.findall(r'nu_(\d+)_(\d+)', col)
    if matches:
        return tuple(map(int, matches[0]))
    return (float('inf'), float('inf'))


def save_new_parquet(breakpoint, original_file):

    """
    Saves new parquet file and columns based on breakpoint.

    Parameters:
    -----------
    breakpoint: str
        Either pre-ms or red-giant (based on criteria for center_h1, Teff, and luminosity)
    original_file: str
        Original filepath for full parquet file.

    Returns:
    --------
    None, but saves new sliced parquet file to directory, to be loaded into memory later.
    """

    print("Original file:", original_file)
    models = pd.read_parquet(filename)

    keep = ['star_mass', 'Yinit', 'Zinit', 'amlt', 'fov0_core', 'fov0_shell', 'star_age', 'center_h1',
        'feh', 'luminosity', 'radius', 'Teff', 'log_g', 'eep']
    keep += [col for col in models.columns if col.startswith('freq_') or col.startswith('inertia_')]

    models = models[keep]

    models['Track'], _ = pd.factorize(models['star_mass'])

    models.rename(columns={'Yinit': 'Y',
                       'Zinit': 'Z',
                       'amlt': 'alpha',
                       'star_mass': 'M',
                       'feh': 'Fe_H'}, inplace=True)

    models.columns = [rename_columns(col) for col in models.columns]
    models.columns = [rename_columns(col, pref1='inertia', pref2='E') for col in models.columns]

    if breakpoint == "pre-ms":
        models = models[np.logical_and(models['center_h1'] > 0.03, models['Teff'] < 7500, models['luminosity'] < 80)]
    if breakpoint == "red-giant":
        models = models[np.logical_and(models['Teff'] < 7500, models['center_h1'] < 0.01)]

    new_parquet_filename = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/pre-run-%s.parquet"%breakpoint
    new_parquet_filename = file_manager.check_and_get_filename(new_parquet_filename)
    models.to_parquet(new_parquet_filename, engine='pyarrow')

    print("Successfully saved new file to:", new_parquet_filename)

def parquet_to_h5(breakpoint):

    """
    A function to transform parquet files to hdf5 files based on pre-defined
    pre-ms or red-giant cutoffs.
    Input:
        breakpoint (str): pre-ms, red-giant
    """
    parquet_file = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/pre-run-%s.parquet"%breakpoint
    print("Original parquet file:", str(parquet_file))
    h5_file = "/gpfs/gibbs/project/bellinger/epb37/research/seistron/data/full_data/pre-run-%s.hdf5"%breakpoint
    parquet_reader = pq.ParquetFile(parquet_file)
    print(f"Total row groups in Parquet: {parquet_reader.num_row_groups}")

    # to create H5 file with proper structure:
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

    print("Successfully saved to:", str(h5_file))


#testing = parquet_to_h5("red-giant")
#print(testing)

