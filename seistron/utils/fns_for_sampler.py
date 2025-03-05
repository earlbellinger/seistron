
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
    """Check if the grid can interpolate across all parameter ranges."""
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


def log_likelihood_old(params, grid_data, observed_values, obs_errors):
    test_point = np.array(params).reshape(1, -1)  # Single point for interpolation
    interpolated_values = linear_interpolator2(grid_data, test_point)[0]
    
    # If interpolation fails (NaNs), return a very low likelihood
    if np.any(np.isnan(interpolated_values)):
        return -np.inf  

    # Compute log-likelihood assuming Gaussian errors
    chi2 = np.sum(((observed_values - interpolated_values) / obs_errors) ** 2)
    return -0.5 * chi2

def log_prior(params, param_bounds):
    for i, (low, high) in enumerate(param_bounds):
        if params[i] < low or params[i] > high:
            return -np.inf  # Reject out-of-bounds samples
    return 0.0  # Uniform prior (valid values)


def log_prior_old(params, param_bounds):
    for p, (low, high) in zip(params, param_bounds):
        if not (low <= p <= high):
            return -np.inf  # Outside bounds
    return 0  # Uniform prior

# Define the full posterior function
def log_posterior_old(params, grid_data, observed_values, obs_errors, param_bounds):
    lp = log_prior(params, param_bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, grid_data, observed_values, obs_errors)


def log_posterior(params, grid_data, observed_values, obs_errors, param_bounds, interp_type="linear"):
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
    ndim = len(param_bounds)  # Number of parameters
    
    # Initialize walkers around a random point within the parameter bounds

    param_bounds = np.array(param_bounds)

    #p0 = np.random.uniform([low for low, high in param_bounds],
    #                    [high for low, high in param_bounds],
    #                    size=(n_walkers, ndim))
    p0_center = (param_bounds[:, 0] + param_bounds[:, 1]) / 2  # Midpoint
    p0_scale = (param_bounds[:, 1] - param_bounds[:, 0]) * 0.1  # Small perturbation

    p0 = np.random.uniform(p0_center - p0_scale, p0_center + p0_scale, size=(n_walkers, ndim))

    for i in range(ndim):
        low, high = param_bounds[i]
        if np.any(p0[:, i] < low) or np.any(p0[:, i] > high):
            print(f"WARNING: Parameter {i} out of bounds in p0!")


    #print("Initial parameters (p0):", p0)
    #print("Min/Max of p0:", np.min(p0, axis=0), np.max(p0, axis=0))

    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, 
                                    args=(grid_data, observed_values, obs_errors, param_bounds, interp_type))
    
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler




# ----- below are functions for the original sampler.py script... -----
"""
def get_likelihood(star_model, data, data_std, datanuids, gridnuids, nukey="nu"):

    '''
    Setup a jifty likelihood using the imported data. The output of the emulator
    is truncated and ordered for nu according to the input data.
    '''


    def ninv(x):
        return x / jft.Vector(data_std)

    # Filter and sort data for nu on grid according to measured nus
    ids = list(np.where((gridnuids == dd).prod(axis=1))[0][0] for dd in datanuids)
    ids = np.array(ids)

    class Response(jft.Model):
        def __init__(self):
            super().__init__(init=star_model.init)

        def __call__(self, x):
            res = star_model(x)
            res[nukey] = res[nukey][ids]
            return jft.Vector(res)

    return jft.Gaussian(data=jft.Vector(data), noise_std_inv=ninv).amend(Response())

def truncated_powerlaw_icdf(z, alpha, xmin, xmax):
    frac = (xmax / xmin) ** (alpha - 1)
    return xmax / ((z + (1.0 - z) * frac) ** (1 / (alpha - 1)))

def myemulator(**kwargs):
    # Placeholder to emulate some dependency of res on x and ensure correct output shape
    res = data.copy()
    res["nu"] *= kwargs["Z"]
    return res

def run_sampler(
    likelihood,
    key,
    N_Samples,
    N_Warmup,
    initial_position=None,
    sampler=blackjax.nuts,
    progress_bar=False,
):
    # Setup the logpdf from jifty
    def logdensity(x):
        return -(likelihood(x) + 0.5 * jft.vdot(x, x))

    if initial_position is None:
        key, init_key = jax.random.split(key)
        initial_position = likelihood.init(init_key)
    # run warmup for mass matrix and stepsize adaptation using stan warmup
    warmup = blackjax.window_adaptation(sampler, logdensity, progress_bar=progress_bar)
    key, warmup_key, sample_key = jax.random.split(key, 3)
    (state, parameters), _ = warmup.run(
        warmup_key, initial_position, num_steps=int(N_Warmup)
    )

    kernel = sampler(logdensity, **parameters).step

    # define sampling loop to run with samplingparameters now fixed
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, params):
            _, rng_key = params
            state, _ = kernel(rng_key, state)
            return state, state

        if progress_bar:
            print("Running NUTS sampler")
            one_step_ = jax.jit(
                blackjax.progress_bar.progress_bar_scan(num_samples)(one_step)
            )
        else:
            one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(
            one_step_, initial_state, (jnp.arange(num_samples), keys)
        )
        return states

    states = inference_loop(sample_key, kernel, state, int(N_Samples))
    # Turn blackjax samples into jifty samples to use tree math for models
    samples = jft.Samples(samples=states.position)
    return samples, states
"""
