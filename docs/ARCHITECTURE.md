# System Architecture

```
Data Sources -> Data Loader -> Feature Engineering -> Forecasting -> Optimization -> Reporting
```

## Components

| Component | Responsibilities |
| --- | --- |
| Data Sources | Orders, order items, inventory reports, items, places, and bill of materials. |
| Data Loader | Standardizes CSV inputs into ML-ready tables. |
| Feature Engineering | Builds time, lag, rolling, price, and calendar features. |
| Forecasting | Predicts daily demand per item and location. |
| Optimization | FEFO prioritization, waste risk scoring, promotion bundling, and prep planning. |
| Reporting | Exports forecasts, recommendations, promotions, and summary metrics. |

## Data Flow

1. Load CSVs from the selected data directory.
2. Aggregate daily sales and attach item and place metadata.
3. Build forecasting features from the daily sales data.
4. Train the forecaster and generate the next-day demand forecast.
5. Align inventory with forecasted items and calculate expiration risk.
6. Generate promotion bundles and prep plans where BOM data is available.
7. Export CSV reports and summary metrics.
