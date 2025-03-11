import pandas as pd

# Load the CSV file
file_path = "/home/ng474/seistron/prelim_data/data_classical.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Output the number of rows
num_rows = len(data)
print(f"Number of rows: {num_rows}")

# Print the first 5 rows
print("First 5 rows:")
print(data.head())
