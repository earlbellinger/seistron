import sys
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
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

from fns_to_read_parquet import extract_numbers
from fns_parquet_data import nu_max, nanloss, save_model, plot_density_results
import file_manager
from file_manager import get_new_filename

"""
A general script to read SLICED .parquet files based on stellar evolution
conditions for ms, red-giant, etc.

As long as the parquet file exists, then this should work (and it does)!

Modified get_model_tracks.py script.

-------------------------------------------------------------------------
Naomi Gluck | Yale University 2024
"""

base_dir = "/home/ng474/seistron/parquets/"

def list_parquet_files(directory):
    """List all Parquet files in the specified directory."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        if files:
            print("Parquet files in the directory:")
            for file in files:
                print(file)
        else:
            print("No Parquet files found in the directory.")
    except Exception as e:
        print(f"Error accessing the directory: {directory}")
        print(e)

# ==================== terminal inputs ======================

if len(sys.argv) < 2:
    print("Usage: python get_best_tracks.py 1) filename (without base_dir or .parquet)")
    print("Files in /parquet directory are:", list_parquet_files(base_dir))

parquet_file_path = sys.argv[1] + ".parquet"
print(".parquet file path:", base_dir + parquet_file_path)

# ===========================================================

df = pd.read_parquet(base_dir + parquet_file_path) # , engine='pyarrow'
df = df.dropna(axis=1, how='all')

for column in df.columns:
    mask = df[column].apply(lambda x: not (isinstance(x, int) or isinstance(x, float)))
    if mask.any():
        print(f"Strings or non-numeric values found in {column}:")
        print(df.loc[mask, column])
        df.loc[mask, column] = np.nan

nu_columns = sorted([col for col in df.columns if col.startswith('nu_')], key=extract_numbers)
other_columns = [col for col in df.columns if col not in nu_columns]
final_columns = other_columns + nu_columns
df = df[final_columns]

print("Check for nans:", np.where(df.isna().all()))

# --------------- adding columns ------------------------

# need to write this up as sys.argv optional

# --------------- train, test, split --------------------

X = df[['M', 'Y', 'Z', 'alpha', 'fov0_core', 'fov0_shell', 'center_h1']]
y = df[[col for col in df.columns if col.startswith('nu_') and col != 'nu_max' or
                                     col in ['Teff', 'L', 'radius', 'Fe_H', 'star_age']]]

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X.values)
y_scaled = y_scaler.fit_transform(y.values)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print("y_train shape:", y_train.shape[1])
print("X_train shape:", X_train.shape[1])

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

print(">>> Creating model... >>>")

model = StellarModel(features=[128, 256, 128, y_train.shape[1]], num_output_channels=8, kernel_size=3)

optimizer = optax.adam(learning_rate=0.001)

params = model.init(jax.random.PRNGKey(0), 
                    jax.random.normal(jax.random.PRNGKey(1), (1, X.shape[1])))

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

print(">>> Initializing training and testing... >>>")

train_losses = []
val_losses = []
epochs = []
best_state = None
min_val_loss = float('inf')

csv_base_dir = '/home/ng474/seistron/parquet_losses/'
csv_filename = get_new_filename(csv_base_dir+'training_log_%s.csv'%(sys.argv[1]))
print("Writing nn losses to:", csv_filename)

def mse_loss(params, inputs, targets):
    predictions = model.apply(params, inputs)
    return nanloss(predictions, targets) #jnp.nanmean((predictions - targets) ** 2)

@jax.jit
def train_step(state, batch):
    inputs, targets = batch
    loss, grads = jax.value_and_grad(mse_loss)(state.params, inputs, targets)
    state = state.apply_gradients(grads=grads)
    return state, loss


with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'New Record'])

# Training loop
for epoch in range(100):
    state, train_loss = train_step(state, (X_train, y_train))
    
    # Prepare row data for CSV
    row = [epoch, train_loss, None, None]
    
    if epoch % 1 == 0:
        val_loss = mse_loss(state.params, X_val, y_val)
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        epochs.append(epoch)
        
        # Update row data with validation loss
        row[2] = val_loss
        
        # Check for new record
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_state = state
            save_model(best_state)
            row[3] = 'New record !!'
        
        # Append row data to CSV file
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

state = best_state
emulator_base_dir = "/home/ng474/seistron/emulators/"
saving_model = save_model(state, get_new_filename(emulator_base_dir+"emulator_%s.pkl"%(sys.argv[1])))

print(">>> Saving Best Models... >>>", saving_model)

y_pred = model.apply(state.params, X_val)
y_val_rescaled = y_scaler.inverse_transform(y_val)
y_pred_rescaled = y_scaler.inverse_transform(y_pred)

# --------------------- figures --------------------------
figure_base_dir = '/home/ng474/seistron/plots/'

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.semilogy()
plt.legend()
plt.savefig(get_new_filename(figure_base_dir + 'MSE_loss.jpg'), dpi = 300)


output_names = y.columns
for i, name in enumerate(output_names):
    if not name.startswith('nu_'):
        plot_density_results(y_val_rescaled[:, i], y_pred_rescaled[:, i], f'{name}', 'Actual', 'Predicted')

print(">>> Density plots done! >>>")

nu_indices = [i for i, name in enumerate(output_names) if name.startswith('nu_')]
mask = ~np.isnan(y_val_rescaled)
y_val_nu = y_val_rescaled[:, nu_indices].flatten()
y_pred_nu = y_pred_rescaled[:, nu_indices].flatten()
plot_freqs = plot_density_results(y_val_nu, y_pred_nu, 'Frequencies', 'Actual', 'Predicted')

print(">>> Frequency plots done! >>>", plot_freqs)

nu_max_indices = [i for i, name in enumerate(output_names) if name.startswith('nu_') and name != 'nu_max']

y_val_nu_max = y_val_rescaled[:, nu_max_indices].flatten()
y_pred_nu_max = y_pred_rescaled[:, nu_max_indices].flatten()

mask_nu_max = ~np.isnan(y_val_nu_max) & ~np.isnan(y_pred_nu_max)

y_val_nu_max_masked = y_val_nu_max[mask_nu_max]
y_pred_nu_max_masked = y_pred_nu_max[mask_nu_max]

plot_masked_freqs = plot_density_results(y_val_nu_max_masked, y_pred_nu_max_masked, 'Frequencies-Max', 'Actual', 'Predicted')

print(">>> Max. Frequency plots done! >>>", plot_masked_freqs)



