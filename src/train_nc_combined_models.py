
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the combined dataset
df = pd.read_csv(
    "data/processed/nc_housing_combined_2024.csv",
    parse_dates=["last_sold_date"],
    low_memory=False
)

# Baseline property features
baseline_features = [
    "beds",
    "baths",
    "sqft",
    "lot_sqft",
    "year_built",
    "lat",
    "lng"
]

# Additional economic features
economic_features = [
    "median_income",
    "population",
    "median_rent",
    "housing_units",
    "annual_price_change",
    "house_price_index",
    "mortgage_rate"
]

enhanced_features = baseline_features + economic_features

target = "last_sold_price"

# Sort sales by date
df = df.sort_values("last_sold_date")

# Use January-September for training and October-December for testing
train_df = df[df["last_sold_date"] < "2024-10-01"]
test_df = df[df["last_sold_date"] >= "2024-10-01"]

print("Training properties:", len(train_df))
print("Testing properties:", len(test_df))

if train_df.empty or test_df.empty:
    raise ValueError("Training or testing dataset is empty.")

os.makedirs("models", exist_ok=True)


def train_and_evaluate(name, features, algorithm, filename):
    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # Missing values are filled using training data only
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{name}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    joblib.dump(pipeline, filename)

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


results = []

# Baseline Linear Regression
results.append(train_and_evaluate(
    "Baseline Linear Regression",
    baseline_features,
    LinearRegression(),
    "models/nc_baseline_linear.pkl"
))

# Baseline Random Forest
results.append(train_and_evaluate(
    "Baseline Random Forest",
    baseline_features,
    RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "models/nc_baseline_forest.pkl"
))

# Enhanced Linear Regression
results.append(train_and_evaluate(
    "Enhanced Linear Regression",
    enhanced_features,
    LinearRegression(),
    "models/nc_enhanced_linear.pkl"
))

# Enhanced Random Forest
results.append(train_and_evaluate(
    "Enhanced Random Forest",
    enhanced_features,
    RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "models/nc_enhanced_forest.pkl"
))

# Save feature lists for the future demo
joblib.dump(
    baseline_features,
    "models/nc_baseline_features.pkl"
)

joblib.dump(
    enhanced_features,
    "models/nc_enhanced_features.pkl"
)

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    "data/processed/nc_model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(results_df.to_string(index=False))
print("\nAll models saved successfully!")
