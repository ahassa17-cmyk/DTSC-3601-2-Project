
import os
import pandas as pd


def combine_data():
    # Load the three datasets
    census = pd.read_csv(
        "data/raw/census_housing_2024.csv",
        dtype={"county_fips": str}
    )

    fhfa = pd.read_csv(
        "data/raw/fhfa_county_2024.csv",
        dtype={"county_fips": str}
    )

    fred = pd.read_csv("data/raw/fred_mortgage_rates.csv")

    # Ensure county FIPS codes have five digits
    census["county_fips"] = census["county_fips"].str.zfill(5)
    fhfa["county_fips"] = fhfa["county_fips"].str.zfill(5)

    # Check for duplicate FHFA county records
    duplicates = fhfa["county_fips"].duplicated().sum()
    print("Duplicate FHFA county codes:", duplicates)

    # Merge Census and FHFA by county FIPS
    combined = pd.merge(
        census,
        fhfa[
            ["county_fips", "annual_price_change", "house_price_index"]
        ],
        on="county_fips",
        how="left",
        validate="one_to_one"
    )

    # Calculate the average mortgage rate for 2024
    fred["date"] = pd.to_datetime(fred["date"])
    fred["mortgage_rate"] = pd.to_numeric(
        fred["mortgage_rate"], errors="coerce"
    )

    fred_2024 = fred[fred["date"].dt.year == 2024]
    average_mortgage_rate = fred_2024["mortgage_rate"].mean()

    # Add the national mortgage rate to each county
    combined["mortgage_rate_2024"] = average_mortgage_rate

    # Check the results
    print("\nTotal counties:", len(combined))
    print(
        "Counties matched with FHFA:",
        combined["house_price_index"].notna().sum()
    )
    print(
        "Counties without FHFA data:",
        combined["house_price_index"].isna().sum()
    )
    print("Average 2024 mortgage rate:", average_mortgage_rate)

    print("\nFirst five rows:")
    print(combined.head())

    # Save the combined dataset
    os.makedirs("data/processed", exist_ok=True)

    combined.to_csv(
        "data/processed/housing_combined_2024.csv",
        index=False
    )

    print("\nCombined dataset saved successfully!")

    return combined


if __name__ == "__main__":
    housing_data = combine_data()
