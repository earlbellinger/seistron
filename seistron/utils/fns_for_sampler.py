
import operator
#import blackjax
#import blackjax.progress_bar
import jax
import jax.numpy as jnp
#import nifty8.re as jft
import numpy as np
from typing import Callable, Union
from functools import partial, reduce
import os, re, sys
import pandas as pd
import emcee 
from fns_for_lininterp import linear_interpolator2
from fns_for_knninterp import knn_interpolator

rng_key = jax.random.PRNGKey(42)

def check_grid_bounds(grid_data, param_bounds):
    """
    Checks whether the given parameter bounds fall within the range of a provided grid.

    Parameters:
    -----------
    grid_data : pandas.DataFrame or array-like
        A dataset representing the grid, where each column corresponds to a parameter.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the allowed bounds for each parameter.

    Returns:
    --------
    bool
        Returns True if all parameter bounds are within the grid range, otherwise False.

    """
    grid_min = np.min(grid_data.to_numpy(), axis=0)  # Ensure NumPy array
    grid_max = np.max(grid_data.to_numpy(), axis=0)

    print("Grid min:", grid_min)  
    print("Grid max:", grid_max)  
    print("Parameter bounds:", param_bounds)  

    for i, (low, high) in enumerate(param_bounds):
        print(f"Checking param {i}: Grid range ({grid_min[i]}, {grid_max[i]}) vs. Bounds ({low}, {high})")
        if grid_min[i] > low or grid_max[i] < high:
            print(f"Warning: Parameter {i} is out of grid range!")
            return False

    return True  # Return True if all params are within bounds


def check_grid_coverage(grid_data, param_bounds, n_samples=100):
    """
    Check if the grid can interpolate across all parameter ranges.

    Parameters:
    -----------
    grid_data : pandas.DataFrame or array-like
        The dataset representing the grid, where each column corresponds to a parameter.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the range for each parameter.
    n_samples : int, optional
        The number of random samples to test within the parameter bounds (default is 100).

    Returns:
    --------
    bool
        Returns True if all sampled parameter points can be interpolated without NaN values,
        otherwise False.
    """
    test_samples = np.random.uniform(
        [low for low, high in param_bounds],
        [high for low, high in param_bounds],
        size=(n_samples, len(param_bounds))
    )

    found_nan = False  # Track if NaNs occur

    for params in test_samples:
        interpolated_values = linear_interpolator2(grid_data, params.reshape(1, -1))
        if np.any(np.isnan(interpolated_values)):
            print(f"NaN encountered in interpolation for params: {params}")
            found_nan = True  # Set flag to True if any NaN is found

    return not found_nan  # Return False if any NaN was encountered




def log_likelihood(params, grid_data, observed_values, obs_errors, param_bounds, interp_type="linear"):
    """
    Computes the log-likelihood of a given parameter set by comparing interpolated
    model predictions with observed values using a chi-squared statistic.

    Parameters:
    -----------
    params : array-like
        A list or array of parameter values for which to evaluate the likelihood.
    grid_data : pandas.DataFrame or array-like
        The dataset representing the model grid, where each row corresponds to a parameter set.
    observed_values : array-like
        The observed data values to be compared against the model.
    obs_errors : array-like
        The uncertainties (errors) associated with each observed value.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the valid range for each parameter.
    interp_type : str, optional
        The interpolation method to use, either "linear" (default) or "knn".
        - "linear": Uses `linear_interpolator2` for interpolation.
        - "knn": Uses `knn_interpolator` with `n_neighbors=20` and distance-based weights.

    Returns:
    --------
    float
        The log-likelihood value, computed as (-0.5 * chi2), where chi2 is the chi-squared statistic.
        If any parameter is out of bounds or the interpolation returns NaN values, (-inf) is returned.

    """
    test_point = np.array(params).reshape(1, -1)  # Single point for interpolation
    for i, (low, high) in enumerate(param_bounds):
        if params[i] < low or params[i] > high:
            print(f"Parameter {i} out of bounds: {params[i]} (should be between {low} and {high})")
    #print("log_likelihood test_point:", test_point, test_point.shape)
    if interp_type == "linear":
        interpolated_values = linear_interpolator2(grid_data, test_point)[0]
    if interp_type == "knn":
        interpolated_values = knn_interpolator(grid_data, test_point, n_neighbors=20, weights='distance')

    if np.any(np.isnan(interpolated_values)):
        print(f"NaN encountered! Interpolated values: {interpolated_values}")
        return -np.inf

    # Compute chi-squared
    chi2 = np.sum(((observed_values - interpolated_values) / obs_errors) ** 2)
    
    # Print debug info for extreme values
    if np.isnan(chi2) or np.isinf(chi2):
        print(f"Invalid chi2: {chi2} for params: {params}")
        print(f"Observed values: {observed_values}")
        print(f"Interpolated values: {interpolated_values}")
        print(f"Errors: {obs_errors}")

    return -0.5 * chi2



def log_prior(params, param_bounds):
    """
    Computes the log-prior probability for a given set of parameters.

    Parameters:
    -----------
    params : array-like
        A list or array of parameter values.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the valid range for each parameter.

    Returns:
    --------
    float
        Returns 0.0 if all parameters are within the specified bounds (uniform prior).
        Returns -inf if any parameter is out of bounds, indicating rejection.
    """
    for i, (low, high) in enumerate(param_bounds):
        if params[i] < low or params[i] > high:
            return -np.inf  # Reject out-of-bounds samples
    return 0.0  # Uniform prior (valid values)


def log_posterior(params, grid_data, observed_values, obs_errors, param_bounds, interp_type="linear"):
    """
    Computes the log-posterior probability for a given set of parameters, combining
    the log-prior and log-likelihood.

    Parameters:
    -----------
    params : array-like
        A list or array of parameter values.
    grid_data : pandas.DataFrame or array-like
        The dataset representing the model grid, where each row corresponds to a parameter set.
    observed_values : array-like
        The observed data values to be compared against the model.
    obs_errors : array-like
        The uncertainties (errors) associated with each observed value.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the valid range for each parameter.
    interp_type : str, optional
        The interpolation method to use, either "linear" (default) or "knn".

    Returns:
    --------
    float
        The log-posterior value, computed as `log_prior + log_likelihood`.
        Returns -inf if the parameters are out of bounds, the likelihood is invalid,
        or the prior is invalid.
    """
    # Ensure all parameters are within bounds
    for i, (low, high) in enumerate(param_bounds):
        if not (low <= params[i] <= high):
            #print(f"Param {i} out of bounds: {params[i]} not in ({low}, {high})")
            return -np.inf  # Reject samples outside bounds

    # Compute log-likelihood
    log_like = log_likelihood(params, grid_data, observed_values, obs_errors, param_bounds, interp_type=interp_type)
    if np.isnan(log_like) or np.isinf(log_like):
        print(f"Invalid log likelihood: {log_like}")
        return -np.inf  # Reject invalid likelihoods

    # Compute log-prior (if using priors)
    log_prior_ = log_prior(params, param_bounds)  
    if np.isnan(log_prior_) or np.isinf(log_prior_):
        print(f"Invalid log prior: {log_prior_}")
        return -np.inf

    log_posterior_value = log_prior_ + log_like
    if np.isnan(log_posterior_value) or np.isinf(log_posterior_value):
        print(f"Invalid log posterior: {log_posterior_value}")
        return -np.inf

    return log_posterior_value



def run_mcmc(grid_data, observed_values, obs_errors, param_bounds, interp_type="linear", n_walkers=50, n_steps=2000):
    """
    Computes the log-posterior probability for a given set of parameters, combining
    the log-prior and log-likelihood.

    Parameters:
    -----------
    params : array-like
        A list or array of parameter values.
    grid_data : pandas.DataFrame or array-like
        The dataset representing the model grid, where each row corresponds to a parameter set.
    observed_values : array-like
        The observed data values to be compared against the model.
    obs_errors : array-like
        The uncertainties (errors) associated with each observed value.
    param_bounds : list of tuples
        A list of (min, max) tuples specifying the valid range for each parameter.
    interp_type : str, optional
        The interpolation method to use, either "linear" (default) or "knn".

    Returns:
    --------
    float
        The log-posterior value, computed as `log_prior + log_likelihood`.
        Returns -inf if the parameters are out of bounds, the likelihood is invalid,
        or the prior is invalid.
    """
    ndim = len(param_bounds)  # Number of parameters
    
    # Initialize walkers around a random point within the parameter bounds

    param_bounds = np.array(param_bounds)

    p0_center = (param_bounds[:, 0] + param_bounds[:, 1]) / 2
    p0_scale = (param_bounds[:, 1] - param_bounds[:, 0]) * 0.05
    p0 = np.random.uniform(p0_center - p0_scale, p0_center + p0_scale, size=(n_walkers, ndim))

    for i in range(ndim):
        low, high = param_bounds[i]
        if np.any(p0[:, i] < low) or np.any(p0[:, i] > high):
            print(f"WARNING: Parameter {i} out of bounds in p0!")

    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, 
                                    args=(grid_data, observed_values, obs_errors, param_bounds, interp_type))
    
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler




