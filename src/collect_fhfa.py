
import pandas as pd


def collect_fhfa_data():
    # Read the FHFA Excel file
    file_path = "data/raw/hpi_at_county.xlsx"
    df = pd.read_excel(file_path, header=5)

    # Select data for 2024
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"] == 2024].copy()

    # Keep relevant columns
    df = df[
        ["State", "County", "FIPS code",
         "Year", "Annual Change (%)", "HPI"]
    ]

    # Rename columns
    df = df.rename(columns={
        "State": "state_name",
        "County": "county_name",
        "FIPS code": "county_fips",
        "Year": "year",
        "Annual Change (%)": "annual_price_change",
        "HPI": "house_price_index"
    })

    # Convert values to numbers
    numeric_columns = [
        "annual_price_change",
        "house_price_index"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Format FIPS codes as five-digit strings
    df["county_fips"] = (
        df["county_fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # Display cleaned data
    print("First five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nTotal records:", len(df))

    # Save cleaned data
    df.to_csv(
        "data/raw/fhfa_county_2024.csv",
        index=False
    )

    print("\nFHFA data saved successfully!")

    return df


if __name__ == "__main__":
    fhfa_data = collect_fhfa_data()
