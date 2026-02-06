"""
Feature engineering for demand forecasting.
"""

from __future__ import annotations

from typing import Optional
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    _HAS_HOLIDAY_CALENDAR = True
except Exception:
    _HAS_HOLIDAY_CALENDAR = False


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "hour_of_day",
    "is_weekend",
    "is_holiday",
    "days_to_holiday",
    "is_ramadan",
    "quantity_lag_1d",
    "quantity_lag_7d",
    "quantity_lag_14d",
    "quantity_lag_28d",
    "quantity_rolling_mean_7d",
    "quantity_rolling_std_7d",
    "quantity_rolling_mean_14d",
    "quantity_rolling_std_14d",
    "quantity_rolling_mean_30d",
    "quantity_rolling_std_30d",
    "quantity_ewm",
    "dow_avg",
    "avg_price",
    "price_change",
    "order_count",
]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not _HAS_HOLIDAY_CALENDAR:
        df["is_holiday"] = 0
        df["days_to_holiday"] = 30
        return df

    min_date = df["date"].min()
    max_date = df["date"].max() + timedelta(days=365)

    calendar = USFederalHolidayCalendar()
    holidays = pd.to_datetime(calendar.holidays(start=min_date, end=max_date)).normalize()
    holiday_set = set(holidays)

    df["is_holiday"] = df["date_day"].isin(holiday_set).astype(int)

    holiday_list = sorted(list(holiday_set))

    def days_to_next(date_value: pd.Timestamp) -> int:
        for holiday in holiday_list:
            if holiday >= date_value:
                return int((holiday - date_value).days)
        return 30

    df["days_to_holiday"] = df["date_day"].apply(days_to_next)
    return df


def build_features(
    daily_sales: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    order_items_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build ML-ready features for demand forecasting.
    """
    df = daily_sales.copy()

    if "date" not in df.columns and "order_date" in df.columns:
        df["date"] = df["order_date"]

    df["date"] = _ensure_datetime(df["date"])
    df = df.sort_values(["place_id", "item_id", "date"])
    df["date_day"] = df["date"].dt.normalize()

    # Order counts from raw order items if available.
    if orders_df is not None and order_items_df is not None:
        orders = orders_df.copy()
        if "order_created_at" not in orders.columns and "created" in orders.columns:
            orders["order_created_at"] = pd.to_datetime(orders["created"], unit="s", errors="coerce")
        if "order_created_at" in orders.columns:
            orders["date_day"] = _ensure_datetime(orders["order_created_at"]).dt.normalize()
            merged = order_items_df.merge(
                orders[["id", "place_id", "date_day"]],
                left_on="order_id",
                right_on="id",
                how="left",
            )
            order_counts = (
                merged.groupby(["place_id", "item_id", "date_day"])["order_id"]
                .nunique()
                .reset_index(name="order_count")
            )
            df = df.merge(order_counts, on=["place_id", "item_id", "date_day"], how="left")

    if "order_count" not in df.columns:
        df["order_count"] = 1
    df["order_count"] = df["order_count"].fillna(0)

    # Price features.
    df["avg_price"] = np.where(df["quantity_sold"] > 0, df["revenue"] / df["quantity_sold"], 0)
    df["avg_price"] = df["avg_price"].replace([np.inf, -np.inf], 0).fillna(0)
    df["price_change"] = (
        df.groupby(["place_id", "item_id"])["avg_price"]
        .pct_change()
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # Time features.
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["hour_of_day"] = 12
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = _add_holiday_features(df)

    # Placeholder for Ramadan seasonality.
    df["is_ramadan"] = 0

    # Lag features.
    grouped = df.groupby(["place_id", "item_id"], sort=False)
    df["quantity_lag_1d"] = grouped["quantity_sold"].shift(1)
    df["quantity_lag_7d"] = grouped["quantity_sold"].shift(7)
    df["quantity_lag_14d"] = grouped["quantity_sold"].shift(14)
    df["quantity_lag_28d"] = grouped["quantity_sold"].shift(28)

    # Rolling features.
    df["quantity_rolling_mean_7d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(7, min_periods=1).mean()
    )
    df["quantity_rolling_std_7d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(7, min_periods=1).std()
    )
    df["quantity_rolling_mean_14d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(14, min_periods=1).mean()
    )
    df["quantity_rolling_std_14d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(14, min_periods=1).std()
    )
    df["quantity_rolling_mean_30d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(30, min_periods=1).mean()
    )
    df["quantity_rolling_std_30d"] = grouped["quantity_sold"].transform(
        lambda series: series.rolling(30, min_periods=1).std()
    )
    df["quantity_ewm"] = grouped["quantity_sold"].transform(
        lambda series: series.ewm(span=7, adjust=False).mean()
    )

    df["dow_avg"] = df.groupby(["place_id", "item_id", "day_of_week"])["quantity_sold"].transform("mean")

    df = df.fillna(0)
    return df
