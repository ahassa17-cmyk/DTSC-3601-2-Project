
import os
import requests
import pandas as pd


def collect_census_data():
    # Census API for 2024 housing data
    url = "https://api.census.gov/data/2024/acs/acs5"

    params = {
        "get": (
            "NAME,B25077_001E,B19013_001E,"
            "B01003_001E,B25058_001E,B25001_001E"
        ),
        "for": "county:*",
        "key": os.environ["CENSUS_API_KEY"].strip()
    }

    # Request data from the API
    response = requests.get(
        url,
        params=params,
        timeout=30,
        allow_redirects=False
    )

    print("Status code:", response.status_code)
    print("Redirect destination:", response.headers.get("Location"))

    if response.status_code in (301, 302, 303, 307, 308):
        raise RuntimeError(
        "Census API redirected the request. "
        "Check whether the API key is valid and activated."
    )

    response.raise_for_status()


    # Convert the response into JSON
    try:
        data = response.json()
    except ValueError:
        print("Error: Census API did not return valid JSON.")
        raise

    # Convert JSON into a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    # Rename columns
    df = df.rename(columns={
        "NAME": "county_name",
        "B25077_001E": "median_home_value",
        "B19013_001E": "median_income",
        "B01003_001E": "population",
        "B25058_001E": "median_rent",
        "B25001_001E": "housing_units"
    })

    # Convert values to numbers
    numeric_columns = [
        "median_home_value",
        "median_income",
        "population",
        "median_rent",
        "housing_units"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        # Handle unavailable Census estimates
        df.loc[df[column] < 0, column] = float("nan")

    # Create county FIPS identifier
    df["county_fips"] = df["state"] + df["county"]

    return df



if __name__ == "__main__":
    housing_data = collect_census_data()

    print(housing_data.head())
    print("Total counties:", len(housing_data))

    # Save the collected data
    housing_data.to_csv(
        "data/raw/census_housing_2024.csv",
        index=False
    )

    print("Census data saved successfully!")

