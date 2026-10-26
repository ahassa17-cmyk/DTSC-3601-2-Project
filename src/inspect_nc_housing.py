
import pandas as pd

# Replace with your actual CSV filename
file_path = "data/raw/nc_housing.csv"

df = pd.read_csv(file_path, low_memory=False)

print("First five rows:")
print(df.head().to_string())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().to_string())
