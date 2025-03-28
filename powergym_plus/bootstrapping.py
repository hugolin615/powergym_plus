import numpy as np
import pandas as pd

# Read the original dataset from CSV
input_file = "13_load_profile.csv"  # Replace with your actual file name
original_data = pd.read_csv(input_file).to_numpy()  # Convert to NumPy array

# Parameters for the new dataset
new_rows = 19584  # Number of rows in the new dataset
new_columns = 6   # Number of columns in the new dataset

# Create new dataset
new_dataset = np.empty((new_rows, new_columns))  # Placeholder for the new dataset


### Create a single load profile which will have columns and rows to create atleast 500 load profile for testing

# Random sampling
for i in range(new_columns):
    # Randomly sample indices from 0 to 8759
    sampled_indices = np.random.choice(original_data.shape[0], size=new_rows, replace=True)
    # Randomly choose one of the three columns (0, 1, or 2)
    sampled_column = np.random.choice(original_data.shape[1])
    # Fill the column in the new dataset
    new_dataset[:, i] = original_data[sampled_indices, sampled_column]

# Convert the result to a DataFrame
new_dataset_df = pd.DataFrame(new_dataset, columns=[f"Column_{i+1}" for i in range(new_columns)])

# Save the new dataset to a CSV file
output_file = "bootstrapped_34_lp.csv"  # Replace with your desired output file name
new_dataset_df.to_csv(output_file, index=False)

print(f"New dataset saved to {output_file}")


### use the following code to create multiple load profile from single load profile

data = pd.read_csv("bootstrapped_34_lp.csv")

for column in data.columns:
    # Create a new DataFrame with just one column
    column_data = data[[column]]
    # Generate a file name for each column
    output_file = f"{column}_output.csv"  # You can modify the naming pattern if needed
    # Save the column data to a CSV file
    column_data.to_csv(output_file, index=False, header=False)
    print(f"Saved {output_file}")

print("All columns have been saved to separate CSV files.")

