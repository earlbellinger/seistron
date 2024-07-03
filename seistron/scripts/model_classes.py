import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from scipy.optimize import curve_fit

import flax
from flax import linen as nn
from flax.training import train_state

import jax
import jax.numpy as jnp
import optax

import seaborn as sns
import re, os

# =================================

class StellarModel(nn.Module):
    features: list
    num_output_channels: int
    kernel_size: int
    
    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        
        x = nn.Conv(features=self.num_output_channels, kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        
        x = nn.Dense(self.features[-1])(x)
        return x
