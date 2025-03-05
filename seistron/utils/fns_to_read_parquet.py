import re


def rename_columns(col, pref1='freq', pref2='nu'):
    # This function uses regex to find patterns and replace them
    match = re.match(pref1 + r"_l(\d+)_n(\d+)", col)
    if match:
        return pref2 + f"_{match.group(1)}_{match.group(2)}"
    else:
        return col

def extract_numbers(col):
    matches = re.findall(r'nu_(\d+)_(\d+)', col)
    if matches:
        return tuple(map(int, matches[0]))
    return (float('inf'), float('inf'))
