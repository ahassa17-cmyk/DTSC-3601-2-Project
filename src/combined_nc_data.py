
import os
import pandas as pd


def format_fips(series):
    # Convert county FIPS values into five-digit strings
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(5)
    )


def combine_nc_data():
    # Load the four datasets
    housing = pd.read_csv(
        "data/processed/nc_housing_cleaned.csv",
        parse_dates=["last_sold_date"],
        low_memory=False
    )

    census = pd.read_csv(
        "data/raw/census_housing_2024.csv"
    )

    fhfa = pd.read_csv(
        "data/raw/fhfa_county_2024.csv"
    )

    fred = pd.read_csv(
        "data/raw/fred_mortgage_rates.csv",
        parse_dates=["date"]
    )

    print("Original housing records:", len(housing))

    # Use 2024 sales to align with the county data
    housing = housing[
        housing["last_sold_date"].dt.year == 2024
    ].copy()

    print("2024 housing sales:", len(housing))

    # Standardize county FIPS identifiers
    for df in [housing, census, fhfa]:
        df["county_fips"] = format_fips(df["county_fips"])

    # Select Census features
    census = census[[
        "county_fips",
        "median_income",
        "population",
        "median_rent",
        "housing_units"
    ]].drop_duplicates(subset=["county_fips"])

    # Select FHFA features
    fhfa = fhfa[[
        "county_fips",
        "annual_price_change",
        "house_price_index"
    ]].drop_duplicates(subset=["county_fips"])

    # Merge Census county information
    combined = housing.merge(
        census,
        on="county_fips",
        how="left",
        validate="many_to_one"
    )

    # Merge FHFA county information
    combined = combined.merge(
        fhfa,
        on="county_fips",
        how="left",
        validate="many_to_one"
    )

    # Calculate 2024 national average mortgage rate
    fred_2024 = fred[
        fred["date"].dt.year == 2024
    ]

    average_mortgage_rate = fred_2024["mortgage_rate"].mean()

    if pd.isna(average_mortgage_rate):
        raise ValueError("No 2024 mortgage rates found.")

    # Match each property sale with the latest available mortgage rate
    fred = fred.sort_values("date").copy()
    combined = combined.sort_values("last_sold_date").copy()

    # Use only mortgage rates available before the sale date
    combined = pd.merge_asof(
        combined,
        fred[["date", "mortgage_rate"]],
        left_on="last_sold_date",
        right_on="date",
        direction="backward"
    )

    # Remove the extra FRED date column
    combined = combined.drop(columns=["date"])

    print(
        "Properties with mortgage rates:",
        combined["mortgage_rate"].notna().sum()
    )

    # Report matching success
    print(
        "Properties matched to Census:",
        combined["median_income"].notna().sum()
    )

    print(
        "Properties matched to FHFA:",
        combined["house_price_index"].notna().sum()
    )

    print("Final combined records:", len(combined))

    # Save combined data
    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/nc_housing_combined_2024.csv"

    combined.to_csv(output_path, index=False)

    print("Combined dataset saved to:", output_path)


if __name__ == "__main__":
    combine_nc_data()
