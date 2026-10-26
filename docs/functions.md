# Code Documentation

## Housing Value Prediction Tool

This document explains the main Python scripts used in the North Carolina Housing Value Prediction Tool.

## 1. Data Collection

### `src/collect_census.py`

Collects county-level housing and demographic information from the U.S. Census Bureau API.

The collected features include median home value, median household income, population, median rent, and housing units.

**Output:** `data/raw/census_housing_2024.csv`

### `src/collect_fred.py`

Retrieves historical 30-year fixed mortgage interest rates from the Federal Reserve Economic Data (FRED) API.

The mortgage rates provide information about borrowing conditions in the housing market.

**Output:** `data/raw/fred_mortgage_rates.csv`

### `src/collect_fhfa.py`

Processes county-level House Price Index data from the Federal Housing Finance Agency.

It extracts the county FIPS code, House Price Index, and annual price changes.

**Output:** `data/raw/fhfa_county_2024.csv`

## 2. Data Preprocessing

### `src/clean_nc_housing.py`

Cleans the original North Carolina property dataset.

The script:

- Filters for single-family homes.
- Keeps residential sales from 2020–2026.
- Removes records missing essential information.
- Filters out clearly unrealistic property values.
- Preserves county identifiers for merging.

**Output:** `data/processed/nc_housing_cleaned.csv`

### `src/combined_nc_data.py`

Combines the North Carolina housing dataset with Census, FHFA, and FRED information.

The script:

- Selects properties sold during 2024.
- Standardizes county FIPS codes.
- Merges county-level Census information.
- Merges county-level FHFA housing price information.
- Matches each property sale to the latest available historical mortgage rate.

**Output:** `data/processed/nc_housing_combined_2024.csv`

## 3. Model Training

### `src/train_nc_combined_models.py`

Trains and evaluates four machine learning models:

1. Baseline Linear Regression
2. Baseline Random Forest
3. Enhanced Linear Regression
4. Enhanced Random Forest

The baseline models use individual property characteristics and geographic location.

The enhanced models also include economic information from Census, FHFA, and FRED.

The script uses January–September 2024 property sales for training and October–December 2024 sales for testing.

Missing feature values are handled with median imputation fitted on the training data.

**Outputs:**

- Trained models in `models/`
- Model comparison results in `data/processed/nc_model_comparison.csv`

## 4. Demonstration

### `bin/run_demo.py`

Provides a small example of the model training process.

The script:

- Loads a sample of 500 North Carolina housing records.
- Splits the data into training and testing sets.
- Trains a Random Forest model.
- Calculates mean absolute error.
- Displays an example predicted price and actual recorded price.

**Run with:**

```bash
python3 bin/run_demo.py
```

## 5. Model Evaluation

The project uses three evaluation metrics:

**Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual sale prices.

**Root Mean Squared Error (RMSE):** Measures prediction error while assigning greater weight to larger mistakes.

**R² Score:** Measures how much variation in property sale prices is explained by the model.

The initial experiments showed that the Enhanced Random Forest achieved the lowest MAE.

## 6. Current Limitations

The project is still in development. The current model uses 2024 property transactions and annual economic information.

Because some annual economic data may not have been available at the time of individual property sales, the current results represent retrospective analysis rather than a fully leakage-free future-price forecast.

The quick-start demo is intended to demonstrate functionality, not to establish final predictive accuracy.