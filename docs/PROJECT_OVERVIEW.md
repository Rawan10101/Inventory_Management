# Fresh Flow Inventory Intelligence

Fresh Flow is an end-to-end demand forecasting and inventory optimization platform for restaurants and grocers. It predicts demand, converts forecasts into prep plans, prioritizes expiring stock, and recommends promotions to reduce waste and stockouts.

## Problem
- Overstocking drives waste and expired inventory.
- Understocking drives stockouts and lost revenue.
- Forecasting is inaccurate when it ignores seasonality and external signals.

## Solution
- Daily, weekly, and monthly demand forecasts at item and location level.
- Prep quantity recommendations based on forecast and shelf life.
- Expiration-aware inventory prioritization using FEFO.
- Promotion and bundle recommendations to move near-expired stock.
- Operational reporting and dashboards.

## Core Outputs
- Item-level demand forecast.
- Inventory risk scores and action recommendations.
- Promotion bundles for at-risk items.
- Prep plan with quantities and cost estimates.

## Data Inputs
- Orders and order items.
- Items catalog and places.
- Inventory reports.
- Bill of materials for prep planning.

See `docs/DEMO_GUIDE.md` for how to run the full pipeline with sample data.
