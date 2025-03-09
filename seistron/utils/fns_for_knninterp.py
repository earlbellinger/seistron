import numpy as np
from scipy.interpolate import LinearNDInterpolator
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import sys
import os


def knn_interpolator(grid_data, test_points, n_neighbors=20, weights='distance'):
    """
    Perform KNN regression to estimate (Teff, L, delta_nu, nu_max) given (M, Y, Z, alpha, fov_core, fov_shell).
    
    Parameters:
        grid_data (pd.DataFrame): A DataFrame containing columns ['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']
        test_points (np.ndarray): Array of shape (N, 6) containing (M, Y, Z, alpha, fov_core, fov_shell) test points
        n_neighbors (int): Number of neighbors to use in KNN
        weights (str): Weighting method for neighbors ('uniform' or 'distance')
    
    Returns:
        np.ndarray: Predicted values of shape (N, 4) for (Teff, luminosity, delta_nu_fit, nu_max)
    """
    # Extract input parameters (features) and output values (targets)
    X_train = grid_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
    y_train = grid_data[['Fe_H','Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

    # Initialize and train the KNN regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)

    # Predict values for test points
    return knn.predict(test_points)

def evaluate_knn_accuracy(grid_data, test_points, true_values, n_neighbors=20, weights='distance'):
    """
    Evaluate the accuracy of the KNN interpolation function using Mean Absolute Error (MAE).

    Parameters:
        grid_data (pd.DataFrame): The dataset used for KNN regression
        test_points (np.ndarray): Array of shape (N, 6) containing test points
        true_values (np.ndarray): Array of shape (N, 4) containing corresponding true values
        n_neighbors (int): Number of neighbors to use in KNN
        weights (str): Weighting method for neighbors ('uniform' or 'distance')

    Returns:
        dict: MAE for each output variable ('Teff', 'luminosity', 'delta_nu_fit', 'nu_max')
    """
    # Ensure inputs are NumPy arrays
    test_points = np.asarray(test_points)
    true_values = np.asarray(true_values)

    # Compute KNN predictions
    predicted_values = knn_interpolator(grid_data, test_points, n_neighbors=n_neighbors, weights=weights)

    # Compute Mean Absolute Error
    mae = {
        'Fe_H': mean_absolute_error(true_values[:, 0], predicted_values[:, 0]),
        'Teff': mean_absolute_error(true_values[:, 1], predicted_values[:, 0]),
        'luminosity': mean_absolute_error(true_values[:, 2], predicted_values[:, 2]),
        'delta_nu_fit': mean_absolute_error(true_values[:, 3], predicted_values[:, 3]),
        'nu_max': mean_absolute_error(true_values[:, 4], predicted_values[:, 4])
    }

    return mae


def knn_grid_search(grid_data, test_points, true_values, neighbor_values=[2, 3, 5, 10, 15, 20], cv_folds=5):
    """
    Perform grid search to find the optimal n_neighbors for kNN interpolation using cross-validation.

    Parameters:
        grid_data (pd.DataFrame): The dataset used for interpolation.
        test_points (numpy.ndarray): Test input points (N, 6).
        true_values (numpy.ndarray): Corresponding true output values (N, 4).
        neighbor_values (list): List of n_neighbors values to test.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: The best n_neighbors value and the corresponding MAE.
    """
    best_n = None
    best_mae = float('inf')
    mae_results = {}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for n_neighbors in neighbor_values:
        fold_maes = []

        for train_idx, val_idx in kf.split(grid_data):
            train_data, val_data = grid_data.iloc[train_idx], grid_data.iloc[val_idx]

            # Extract training input and output
            train_X = train_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
            train_Y = train_data[['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

            # Extract validation input and output
            val_X = val_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
            val_Y = val_data[['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']].values

            # Train kNN model
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn.fit(train_X, train_Y)

            # Predict on validation set
            val_predictions = knn.predict(val_X)

            # Compute MAE for this fold
            fold_mae = mean_absolute_error(val_Y, val_predictions)
            fold_maes.append(fold_mae)

        # Average MAE across folds
        avg_mae = np.mean(fold_maes)
        mae_results[n_neighbors] = avg_mae

        # Update best parameters
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_n = n_neighbors

    return {"best_n_neighbors": best_n, "best_mae": best_mae, "all_mae_results": mae_results}


def knn_grid_search2(grid_data, neighbor_values=[2, 3, 5, 10, 15, 20], cv_folds=5):
    """
    Perform grid search to find the optimal n_neighbors for kNN interpolation using cross-validation.

    Parameters:
        grid_data (pd.DataFrame): The dataset used for interpolation.
        neighbor_values (list): List of n_neighbors values to test.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: The best n_neighbors value and corresponding MAE for each output variable.
    """
    target_columns = ['Fe_H', 'Teff', 'luminosity', 'delta_nu_fit', 'nu_max']  # Explicitly define targets
    best_n = {col: None for col in target_columns}
    best_mae = {col: float('inf') for col in target_columns}
    mae_results = {n: {col: [] for col in target_columns} for n in neighbor_values}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for n_neighbors in neighbor_values:
        fold_maes = {col: [] for col in target_columns}

        for train_idx, val_idx in kf.split(grid_data):
            train_data, val_data = grid_data.iloc[train_idx], grid_data.iloc[val_idx]

            # Extract training input and output
            train_X = train_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
            train_Y = train_data[target_columns].values  # Select target columns explicitly

            # Extract validation input and output
            val_X = val_data[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell']].values
            val_Y = val_data[target_columns].values  # Select target columns explicitly

            # Train kNN model
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn.fit(train_X, train_Y)

            # Predict on validation set
            val_predictions = knn.predict(val_X)

            # Compute MAE for each output variable separately
            for i, col in enumerate(target_columns):
                mae_value = mean_absolute_error(val_Y[:, i], val_predictions[:, i])
                fold_maes[col].append(mae_value)

        # Average MAE across folds
        avg_mae = {col: np.mean(fold_maes[col]) for col in target_columns}
        mae_results[n_neighbors] = avg_mae

        # Update best parameters for each variable
        for col in target_columns:
            if avg_mae[col] < best_mae[col]:
                best_mae[col] = avg_mae[col]
                best_n[col] = n_neighbors  # Store best n_neighbors for each variable

    return {"best_n_neighbors": best_n, "best_mae": best_mae, "all_mae_results": mae_results}


