
import pandas as pd

df = pd.read_csv(
    "data/processed/nc_housing_combined_2024.csv",
    low_memory=False
)

print("Total properties:", len(df))

print("\nMissing values:")
print(df.isnull().sum().to_string())

print("\nDuplicate property IDs:")
print(df["id"].duplicated().sum())

print("\nSale price statistics:")
print(df["last_sold_price"].describe())

print("\nSale date range:")
print(df["last_sold_date"].min())
print(df["last_sold_date"].max())
