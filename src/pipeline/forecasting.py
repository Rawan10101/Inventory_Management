"""
Forecasting utilities with a baseline model and optional advanced model.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from feature_engineering import FEATURE_COLUMNS


class BaselineForecaster:
    """
    Baseline demand forecaster using RandomForestRegressor.
    """

    def __init__(self, forecast_horizon: str = "daily"):
        self.forecast_horizon = forecast_horizon
        self.model: Optional[RandomForestRegressor] = None
        self.feature_columns = FEATURE_COLUMNS

    def train(self, df, target_col: str = "quantity_sold"):
        print(f"Training baseline {self.forecast_horizon} forecaster...")
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()

        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, df) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        X = df[self.feature_columns].copy()
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    def evaluate(self, df, target_col: str = "quantity_sold") -> Dict[str, float]:
        X = df[self.feature_columns].copy()
        y_true = df[target_col].copy()

        valid_idx = ~y_true.isna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]

        y_pred = self.predict(df.loc[valid_idx])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        print("Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        return metrics


def get_forecaster(forecast_horizon: str = "daily", prefer_advanced: bool = False):
    if prefer_advanced:
        try:
            from demand_forecasting import DemandForecaster

            print("Using advanced DemandForecaster")
            return DemandForecaster(forecast_horizon=forecast_horizon)
        except Exception as exc:
            print(f"Advanced forecaster unavailable: {exc}")

    print("Using baseline RandomForest forecaster")
    return BaselineForecaster(forecast_horizon=forecast_horizon)
