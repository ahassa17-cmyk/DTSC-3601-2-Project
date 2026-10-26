import os
import requests
import pandas as pd


def collect_fred_data():
    # FRED API endpoint
    url = "https://api.stlouisfed.org/fred/series/observations"

    # Parameters for mortgage rate data
    params = {
        "series_id": "MORTGAGE30US",
        "api_key": os.environ["FRED_API_KEY"].strip(),
        "file_type": "json",
        "observation_start": "2020-01-01",
        "observation_end": "2024-12-31"
    }

    # Request data from FRED
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    # Convert JSON into a DataFrame
    data = response.json()
    df = pd.DataFrame(data["observations"])

    # Keep the columns we need
    df = df[["date", "value"]]

    # Rename columns
    df = df.rename(columns={
        "value": "mortgage_rate"
    })

    # Convert columns to appropriate types
    df["date"] = pd.to_datetime(df["date"])
    df["mortgage_rate"] = pd.to_numeric(
        df["mortgage_rate"], errors="coerce"
    )

    return df



if __name__ == "__main__":
    mortgage_data = collect_fred_data()

    print(mortgage_data.head())
    print("Total observations:", len(mortgage_data))

    # Save data as CSV
    mortgage_data.to_csv(
        "data/raw/fred_mortgage_rates.csv",
        index=False
    )

    print("FRED data saved successfully!")
