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
