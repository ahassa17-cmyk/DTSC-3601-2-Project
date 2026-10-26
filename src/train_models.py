
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_models():
    # Load the combined housing dataset
    df = pd.read_csv(
        "data/processed/housing_combined_2024.csv"
    )

    # Select the features used for prediction
    features = [
        "median_income",
        "population",
        "median_rent",
        "housing_units"
    ]

    target = "median_home_value"

    # Keep only rows with complete data
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    print("Counties used for training:", len(df))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Train Random Forest Regression
    forest_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    forest_model.fit(X_train, y_train)


    # Display Random Forest feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": forest_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nRandom Forest Feature Importance:")
    print(importance.to_string(index=False))


    # Evaluate both models
    models = {
        "Linear Regression": linear_model,
        "Random Forest": forest_model
    }

    for name, model in models.items():
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test, predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")


if __name__ == "__main__":
    train_models()
