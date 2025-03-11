
"""
These functions were used with the parquet data when at some point 
I was using a NN for interpolation. I don't think these are used at 
the moment, so I'll leave off writing documentation for them
for now...
"""


import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import seaborn as sns
from scipy.optimize import curve_fit

import flax
from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp
import optax

import file_manager
from file_manager import get_new_filename

def nu_max(M, R, T_eff):

    M_sun = 1
    R_sun = 1
    T_eff0 = 5777
    nu_max0 = 3090 # mu-Hz

    nu_max_val = ((M/M_sun) * (R/R_sun)**-2 * (T_eff/T_eff0)**-1/2)*nu_max0

    return nu_max_val

def nanloss(pred, targets):
    mask = jnp.isnan(pred) | jnp.isnan(targets)
    pred = jnp.where(mask, 0, pred)
    targets = jnp.where(mask, 0, targets)
    return jnp.mean((pred - targets) ** 2, where=~mask)

# Define loss function
def mse_loss(params, inputs, targets):
    predictions = model.apply(params, inputs)
    return nanloss(predictions, targets) #jnp.nanmean((predictions - targets) ** 2)

# Example training step
@jax.jit
def train_step(state, batch):
    inputs, targets = batch
    loss, grads = jax.value_and_grad(mse_loss)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss

def save_model(state, filename="emulator-noname.pkl"):
    with open(filename, 'wb') as f:
        serialized_state = flax.serialization.to_bytes(state)
        f.write(serialized_state)


def plot_density_results(y_true, y_pred, title, xlabel, ylabel, figure_base_dir = "/home/ng474/seistron/plots/"):
    y_true = y_true[::1000]
    y_pred = y_pred[::1000]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Density plot for predictions vs. true values on the first subplot
    sns.kdeplot(x=y_true, y=y_pred, cmap="Blues", fill=True, ax=ax1)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.suptitle(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Density plot for residuals on the second subplot
    sns.kdeplot(x=y_true, y=residuals, cmap="Blues", fill=True, ax=ax2)
    ax2.axhline(0, color='k', linestyle='--', linewidth=2)
    #ax2.set_title('Residuals')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Residuals')
    
    # Calculate median and median absolute deviation (MAD) for residuals
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    ylims = median + 5 * mad * np.array([-1, 1])
    #ax2.set_ylim(ylims) # NG commented out 6/26/24 due to nans/infs issue
    
    # Display the plot
    plt.tight_layout()
    fig_filename = figure_base_dir + "%s_plot.jpg"%title
    fig_filename_ = get_new_filename(fig_filename)
    print("Figure saved as:", fig_filename_)
    plt.savefig(fig_filename_, bbox_inches= 'tight', dpi=300)
