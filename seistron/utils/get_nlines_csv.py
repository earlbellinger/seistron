import pandas as pd

# Load the CSV file
input_file = "/home/ng474/seistron/prelim_data/data_classical.csv"
output_file = "/home/ng474/seistron/prelim_data/data_classical_first_5000_rows.csv"

# Read the first 5000 rows
data = pd.read_csv(input_file, nrows=5000)

# Save the selected rows to a new CSV file
data.to_csv(output_file, index=False)

print(f"The first 5000 rows have been saved to {output_file}")

