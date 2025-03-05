import sys
import pyarrow.parquet as pq
import pyarrow.csv as pc
import numpy as np

# Define input/output file paths
base_dir = "/home/ng474/seistron/prelim_data/"
input_file = base_dir + "ms-pi.parquet"

# Check command-line arguments
if len(sys.argv) < 3:
    print("Usage: python slice_parquet.py <number of rows> <mode: first/random>")
    sys.exit(1)

# Parse arguments
nrows = int(sys.argv[1])
mode = sys.argv[2].lower()

# Read the Parquet file
table = pq.read_table(input_file, columns=None, use_threads=True)
num_rows = table.num_rows  # Get total number of rows

# Ensure we don't request more rows than available
nrows = min(nrows, num_rows)

output_file = f"{base_dir}ms-pi_{mode}_{nrows}_rows.csv"

if mode == "first":
    subset_table = table.slice(0, nrows)
elif mode == "random":
    random_indices = np.random.choice(num_rows, nrows, replace=False)
    subset_table = table.take(random_indices)
else:
    print("Invalid mode. Use 'first' or 'random'.")
    sys.exit(1)

# Convert the subset table to a CSV file
pc.write_csv(subset_table, output_file)

print(f"{nrows} {'randomly selected' if mode == 'random' else 'first'} rows have been saved to {output_file}.")

