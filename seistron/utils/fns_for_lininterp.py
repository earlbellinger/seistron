import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.metrics import mean_absolute_error
import sys
import os



def linear_interpolator2(grid_data, test_points):
    """
    Perform linear interpolation for multiple points.

    Parameters:
        grid_data (pd.DataFrame): A DataFrame containing columns ['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell', 'Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']
        test_points (np.ndarray): An array of shape (N, 6) with columns corresponding to (M, Y, Z, alpha, fov_core, fov_shell).

    Returns:
        np.ndarray: Interpolated values of shape (N, 4) corresponding to (Teff, luminosity, delta_nu_fit, nu_max)
    """
    # Extract input parameters and output values from the grid
    points = grid_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
    values = grid_data[['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

    # Create interpolator
    #interpolator = LinearNDInterpolator(points, values)
    interpolator = NearestNDInterpolator(points, values)

    # Perform interpolation on all test points
    results = interpolator(test_points)

    # Replace None (out-of-bounds points) with NaNs
    return np.where(results == None, np.nan, results)



def evaluate_interp_accuracy2(grid_data, test_points, true_values):
    """
    Evaluate the accuracy of the interpolation function using Mean Absolute Error (MAE).
    
    Parameters:
        grid_data (pd.DataFrame): The dataset used for interpolation
        test_points (np.ndarray): Array of shape (N, 6) containing (M, Y, Z, alpha, fov_core, fov_shell) test points
        true_values (np.ndarray): Array of shape (N, 4) containing corresponding true (Teff, luminosity, delta_nu_fit, nu_max) values
    
    Returns:
        dict: MAE for each output variable ('Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max')
    """
    # Ensure inputs are NumPy arrays
    test_points = np.asarray(test_points)
    true_values = np.asarray(true_values)

    # Compute interpolated values in a vectorized manner
    interpolated_values = linear_interpolator2(grid_data, test_points)

    # Handle cases where interpolation returned NaNs
    valid_mask = ~np.isnan(interpolated_values).any(axis=1)

    if not np.any(valid_mask):
        raise ValueError("All interpolated values are NaN. Check test points and interpolation grid.")

    # Compute Mean Absolute Error only for valid points
    mae = {
        'Fe_H': mean_absolute_error(true_values[valid_mask, 0], interpolated_values[valid_mask, 0]),
        'Teff': mean_absolute_error(true_values[valid_mask, 1], interpolated_values[valid_mask, 1]),
        'luminosity': mean_absolute_error(true_values[valid_mask, 2], interpolated_values[valid_mask, 2]),
        'delta_nu_fit': mean_absolute_error(true_values[valid_mask, 3], interpolated_values[valid_mask, 3]),
        'nu_max': mean_absolute_error(true_values[valid_mask, 4], interpolated_values[valid_mask, 4])
    }

    return mae




