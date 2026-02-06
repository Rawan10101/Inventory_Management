# Modeling Approach

## Forecasting Features

| Feature | Description |
| --- | --- |
| day_of_week | Day of week derived from date. |
| day_of_month | Day of month derived from date. |
| week_of_year | ISO week number. |
| month | Month number. |
| quarter | Quarter number. |
| hour_of_day | Fixed to midday for daily data. |
| is_weekend | Weekend flag. |
| is_holiday | US federal holiday flag. |
| days_to_holiday | Days until next holiday. |
| is_ramadan | Placeholder flag, set to 0 by default. |
| quantity_lag_1d | 1-day lag of quantity sold. |
| quantity_lag_7d | 7-day lag of quantity sold. |
| quantity_lag_14d | 14-day lag of quantity sold. |
| quantity_lag_28d | 28-day lag of quantity sold. |
| quantity_rolling_mean_7d | 7-day rolling mean. |
| quantity_rolling_std_7d | 7-day rolling standard deviation. |
| quantity_rolling_mean_14d | 14-day rolling mean. |
| quantity_rolling_std_14d | 14-day rolling standard deviation. |
| quantity_rolling_mean_30d | 30-day rolling mean. |
| quantity_rolling_std_30d | 30-day rolling standard deviation. |
| quantity_ewm | Exponentially weighted mean. |
| dow_avg | Average quantity for the same day-of-week. |
| avg_price | Daily average price. |
| price_change | Day-over-day price change. |
| order_count | Number of orders containing the item. |

## Forecasting Models

- Baseline model: RandomForestRegressor using the engineered features.
- Advanced model: Stacking ensemble in `src/models/demand_forecasting.py` when XGBoost and LightGBM are available.
- Evaluation metrics: MAE, RMSE, and MAPE with time-aware validation.

## Expiration Risk Scoring

Risk score is a weighted composite of three signals.
- Expiry urgency based on days until expiration.
- Overstocking risk based on days of inventory versus days to expiry.
- Value at risk based on unit cost and quantity on hand.

## Promotion Optimization

Promotion bundles are generated from co-purchase patterns using order history. Near-expiry items are paired with complementary items to increase sell-through while preserving margin.

## Prep Planning

Prep quantities are calculated from forecasted demand and bill of materials. Prep is capped by ingredient shelf life and current stock.
