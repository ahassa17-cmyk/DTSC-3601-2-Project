
import pandas as pd
import os


def clean_nc_housing():
    # Replace with your actual CSV filename
    file_path = "data/raw/nc_housing.csv"

    # Load the dataset
    df = pd.read_csv(file_path, low_memory=False)

    print("Original properties:", len(df))

    # Keep single-family homes in North Carolina
    df = df[
        (df["state"] == "NC") &
        (df["property_type"] == "single_family")
    ].copy()

    # Convert sale dates into datetime values
    df["last_sold_date"] = pd.to_datetime(
        df["last_sold_date"],
        errors="coerce"
    )

    # Keep recent sales
    df = df[
        (df["last_sold_date"] >= "2020-01-01") &
        (df["last_sold_date"] <= "2026-10-26")
    ].copy()

    # Select columns for the model
    columns = [
        "id",
        "county_fips",
        "beds",
        "baths",
        "sqft",
        "lot_sqft",
        "year_built",
        "zip",
        "lat",
        "lng",
        "last_sold_date",
        "last_sold_price"
    ]

    df = df[columns].copy()

    # Convert numeric columns
    numeric_columns = [
        "beds",
        "baths",
        "sqft",
        "lot_sqft",
        "year_built",
        "lat",
        "lng",
        "last_sold_price"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with missing essential values
    df = df.dropna(subset=[
        "county_fips",
        "beds",
        "baths",
        "sqft",
        "year_built",
        "lat",
        "lng",
        "last_sold_price"
    ])

    # Remove clearly unrealistic values
    df = df[
        (df["beds"].between(1, 10)) &
        (df["baths"].between(1, 10)) &
        (df["sqft"].between(400, 15000)) &
        (df["year_built"].between(1800, 2026)) &
        (df["last_sold_price"].between(30000, 10000000))
    ].copy()

    # Save the cleaned dataset
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        "data/processed/nc_housing_cleaned.csv",
        index=False
    )

    print("Cleaned properties:", len(df))
    print("Cleaned data saved successfully!")

    return df


if __name__ == "__main__":
    clean_nc_housing()
