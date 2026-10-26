
import os
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_nc_models():
    # Load cleaned North Carolina housing data
    df = pd.read_csv(
        "data/processed/nc_housing_cleaned.csv",
        parse_dates=["last_sold_date"]
    )

    # Select features
    features = [
        "beds",
        "baths",
        "sqft",
        "lot_sqft",
        "year_built",
        "lat",
        "lng"
    ]

    target = "last_sold_price"

    # Remove rows missing model inputs
    df = df.dropna(subset=features + [target, "last_sold_date"])

    # Use older sales for training
    train_df = df[
        df["last_sold_date"] < "2025-01-01"
    ].copy()

    # Use newer sales for testing
    test_df = df[
        df["last_sold_date"] >= "2025-01-01"
    ].copy()

    print("Training properties:", len(train_df))
    print("Testing properties:", len(test_df))

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Not enough data in the training or testing period."
        )

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train Random Forest
    forest_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    forest_model.fit(X_train, y_train)

    # Evaluate both models
    models = {
        "Linear Regression": linear_model,
        "Random Forest": forest_model
    }

    for name, model in models.items():
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")

    # Save the models
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        linear_model,
        "models/nc_linear_regression.pkl"
    )

    joblib.dump(
        forest_model,
        "models/nc_random_forest.pkl"
    )

    # Save the feature names for prediction
    joblib.dump(
        features,
        "models/nc_features.pkl"
    )

    print("\nNorth Carolina models saved successfully!")


if __name__ == "__main__":
    train_nc_models()
