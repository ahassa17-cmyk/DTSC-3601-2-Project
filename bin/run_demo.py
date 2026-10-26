
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Find the project folder
project_dir = Path(__file__).resolve().parent.parent

# Load the small example dataset
data_path = project_dir / "data/processed/nc_housing_sample.csv"
df = pd.read_csv(data_path)

# Features used for the demonstration
features = [
    "beds",
    "baths",
    "sqft",
    "lot_sqft",
    "year_built",
    "lat",
    "lng",
    "median_income",
    "population",
    "median_rent",
    "housing_units",
    "annual_price_change",
    "house_price_index",
    "mortgage_rate"
]

X = df[features]
y = df["last_sold_price"]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a small Random Forest model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("forest", RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("North Carolina Housing Prediction Demo")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Mean Absolute Error: ${mae:,.2f}")

# Show an example prediction
print(f"Example predicted price: ${predictions[0]:,.2f}")
print(f"Actual recorded price: ${y_test.iloc[0]:,.2f}")
