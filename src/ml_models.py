"""
Machine learning module for monthly expense prediction.

Trains and compares real scikit-learn models (Linear Regression,
Random Forest, Gradient Boosting) on the user's own monthly expense
history using lagged/rolling features. If there isn't enough data for
meaningful supervised learning, falls back to a transparent statistical
baseline (weighted moving average) and clearly labels it as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.preprocessing import build_ml_feature_table

MIN_ROWS_FOR_ML = 6  # minimum feature rows required to attempt real ML training
FEATURE_COLUMNS = ["time_idx", "expense_lag1", "expense_lag2", "income_lag1", "rolling_mean_3"]
TARGET_COLUMN = "Expense"


@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: Optional[float]


@dataclass
class ExpensePredictionResult:
    used_ml: bool
    best_model_name: str
    predicted_next_month: float
    model_results: Dict[str, ModelResult]
    message: str


def _evaluate_model(model, X_train, X_test, y_train, y_test) -> ModelResult:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds) if len(y_test) >= 2 else None
    return ModelResult(name=model.__class__.__name__, mae=mae, rmse=rmse, r2=r2)


def predict_next_month_expense(monthly_df: pd.DataFrame) -> ExpensePredictionResult:
    """
    Train and compare regression models on monthly expense history, then
    predict next month's expense using the best-performing model.
    Falls back to a weighted moving average when data is too small.
    """
    if monthly_df is None or monthly_df.empty:
        return ExpensePredictionResult(
            used_ml=False,
            best_model_name="None",
            predicted_next_month=0.0,
            model_results={},
            message="No transaction data available to generate a prediction.",
        )

    feature_table = build_ml_feature_table(monthly_df)

    if len(feature_table) < MIN_ROWS_FOR_ML:
        # Fallback: weighted moving average baseline, using more weight on recent months
        recent = monthly_df["Expense"].tail(3).values
        if len(recent) == 0:
            baseline = 0.0
        else:
            weights = np.arange(1, len(recent) + 1)
            baseline = float(np.average(recent, weights=weights))
        return ExpensePredictionResult(
            used_ml=False,
            best_model_name="Weighted Moving Average (baseline)",
            predicted_next_month=round(baseline, 2),
            model_results={},
            message=(
                "Your dataset has too few months of history for reliable machine learning "
                "training (at least a few months of data are recommended). "
                "A statistical baseline (weighted moving average of recent months) was used instead."
            ),
        )

    X = feature_table[FEATURE_COLUMNS]
    y = feature_table[TARGET_COLUMN]

    test_size = 0.25 if len(feature_table) >= 8 else 1 / len(feature_table)
    test_size = min(max(test_size, 1 / len(feature_table)), 0.4)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        if len(X_test) == 0:
            X_train, X_test = X.iloc[:-1], X.iloc[-1:]
            y_train, y_test = y.iloc[:-1], y.iloc[-1:]
    except ValueError:
        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y.iloc[:-1], y.iloc[-1:]

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.08, random_state=42),
    }

    results: Dict[str, ModelResult] = {}
    for name, model in candidate_models.items():
        try:
            results[name] = _evaluate_model(model, X_train, X_test, y_train, y_test)
        except Exception:  # noqa: BLE001 - a single model failing shouldn't break the app
            continue

    if not results:
        recent = monthly_df["Expense"].tail(3).values
        baseline = float(np.mean(recent)) if len(recent) else 0.0
        return ExpensePredictionResult(
            used_ml=False,
            best_model_name="Simple Average (baseline)",
            predicted_next_month=round(baseline, 2),
            model_results={},
            message="Model training was not successful; used a simple average of recent months instead.",
        )

    # Best model = lowest MAE
    best_name = min(results, key=lambda n: results[n].mae)
    best_model = candidate_models[best_name]

    # Refit best model on ALL available data for the final forecast
    best_model.fit(X, y)

    last_row = feature_table.iloc[-1]
    next_features = pd.DataFrame([{
        "time_idx": last_row["time_idx"] + 1,
        "expense_lag1": last_row["Expense"],
        "expense_lag2": last_row["expense_lag1"],
        "income_lag1": last_row["Income"],
        "rolling_mean_3": np.mean(monthly_df["Expense"].tail(3)),
    }])[FEATURE_COLUMNS]

    prediction = float(best_model.predict(next_features)[0])
    prediction = max(0.0, prediction)

    return ExpensePredictionResult(
        used_ml=True,
        best_model_name=best_name,
        predicted_next_month=round(prediction, 2),
        model_results=results,
        message=f"Trained and compared {len(results)} regression model(s) on {len(feature_table)} months of data.",
    )
