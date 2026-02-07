"""
Pipeline runner for the Fresh Flow project.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import pandas as pd

# Ensure models directory is importable.
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "src" / "models"
PIPELINE_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))

from data_loader import DataLoader
from expiration_manager import ExpirationManager, PromotionOptimizer
from prep_calculator import PrepCalculator

from feature_engineering import build_features
from forecasting import get_forecaster


def _normalize_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "title" in df.columns and "item_name" not in df.columns:
        df = df.rename(columns={"title": "item_name"})
    if "title_place" in df.columns and "merchant_name" not in df.columns:
        df = df.rename(columns={"title_place": "merchant_name"})
    if "merchant_name" not in df.columns:
        df["merchant_name"] = "Unknown"
    return df


def _prepare_inventory(inventory_df: pd.DataFrame) -> pd.DataFrame:
    if inventory_df is None or inventory_df.empty:
        return pd.DataFrame()

    inv = inventory_df.copy()

    if "item_id" not in inv.columns and "id" in inv.columns:
        inv["item_id"] = inv["id"]

    if "quantity_on_hand" not in inv.columns:
        if "current_stock" in inv.columns:
            inv["quantity_on_hand"] = inv["current_stock"]
        elif "quantity" in inv.columns:
            inv["quantity_on_hand"] = inv["quantity"]
        else:
            inv["quantity_on_hand"] = 0

    if "item_name" not in inv.columns:
        if "title" in inv.columns:
            inv["item_name"] = inv["title"]
        else:
            inv["item_name"] = "Unknown"

    if "unit_cost" not in inv.columns:
        if "price" in inv.columns:
            inv["unit_cost"] = inv["price"] * 0.6
        else:
            inv["unit_cost"] = 10.0

    if "total_value" not in inv.columns:
        inv["total_value"] = inv["quantity_on_hand"] * inv["unit_cost"]

    if "days_until_expiration" not in inv.columns:
        rng = np.random.default_rng(42)
        inv["days_until_expiration"] = rng.integers(1, 15, size=len(inv))

    if "report_date" not in inv.columns:
        inv["report_date"] = pd.Timestamp.utcnow().normalize()

    return inv


def _prepare_bom(bom_df: pd.DataFrame) -> pd.DataFrame:
    if bom_df is None or bom_df.empty:
        return pd.DataFrame()

    bom = bom_df.copy()

    if "menu_item_id" not in bom.columns and "parent_sku_id" in bom.columns:
        bom = bom.rename(columns={"parent_sku_id": "menu_item_id"})

    if "ingredient_id" not in bom.columns and "sku_id" in bom.columns:
        bom = bom.rename(columns={"sku_id": "ingredient_id"})

    if "quantity_per_serving" not in bom.columns:
        if "ingredient_quantity" in bom.columns:
            bom = bom.rename(columns={"ingredient_quantity": "quantity_per_serving"})
        elif "quantity" in bom.columns:
            bom = bom.rename(columns={"quantity": "quantity_per_serving"})

    required_cols = [
        "menu_item_id",
        "ingredient_id",
        "ingredient_name",
        "quantity_per_serving",
        "stock_unit",
        "unit_cost",
        "shelf_life_days",
    ]

    for col in required_cols:
        if col not in bom.columns:
            if col == "ingredient_name":
                bom[col] = "Unknown"
            elif col == "stock_unit":
                bom[col] = "unit"
            elif col == "unit_cost":
                bom[col] = 0.0
            elif col == "shelf_life_days":
                bom[col] = 3
            else:
                bom[col] = 0

    return bom


def run_pipeline(
    data_dir: str,
    output_dir: str,
    forecast_horizon: str = "daily",
    prefer_advanced: bool = False,
    prep_horizon_days: int = 1,
) -> Dict[str, object]:
    print("=" * 60)
    print("FRESH FLOW FULL PIPELINE")
    print("=" * 60)

    loader = DataLoader(data_dir)

    orders = loader.load_orders()
    order_items = loader.load_order_items()
    daily_sales = loader.prepare_daily_sales()
    daily_sales = _normalize_daily_sales(daily_sales)

    if daily_sales.empty:
        raise ValueError("No sales data available after preprocessing.")

    features_df = build_features(daily_sales, orders, order_items)

    forecaster = get_forecaster(forecast_horizon=forecast_horizon, prefer_advanced=prefer_advanced)
    forecaster.train(features_df, target_col="quantity_sold")
    model_metrics = forecaster.evaluate(features_df, target_col="quantity_sold")

    last_date = features_df["date"].max()
    latest_rows = features_df[features_df["date"] == last_date].copy()

    predictions = forecaster.predict(latest_rows)

    forecast_df = latest_rows[["item_id", "item_name", "place_id", "merchant_name"]].copy()
    forecast_df["forecast_date"] = last_date + timedelta(days=1)
    forecast_df["predicted_daily_demand"] = np.maximum(predictions, 0).round(2)

    inventory_full = loader.prepare_inventory_snapshot()
    inventory_full = _prepare_inventory(inventory_full)

    inventory_for_expiration = inventory_full.copy()
    if not inventory_for_expiration.empty:
        inventory_for_expiration = inventory_for_expiration[
            inventory_for_expiration["item_id"].isin(forecast_df["item_id"])
        ]

    expiration_manager = ExpirationManager(forecaster)

    if inventory_for_expiration.empty:
        prioritized = pd.DataFrame()
        recommendations = pd.DataFrame()
        at_risk = pd.DataFrame()
    else:
        prioritized = expiration_manager.prioritize_inventory(inventory_for_expiration, forecast_df)
        recommendations = expiration_manager.recommend_actions(prioritized)
        at_risk = recommendations[recommendations["risk_category"].isin(["critical", "high", "medium"])]

    promotions = []
    if not at_risk.empty:
        promo_optimizer = PromotionOptimizer(order_items)
        promotions = promo_optimizer.create_bundle_promotions(at_risk)

    bom_df = loader.load_bill_of_materials()
    bom_df = _prepare_bom(bom_df)

    prep_plan = pd.DataFrame()
    if not bom_df.empty:
        prep_calculator = PrepCalculator(bom_df, inventory_full)
        prep_plan = prep_calculator.calculate_prep_quantities(
            forecast_df,
            prep_date=last_date + timedelta(days=1),
            prep_horizon_days=prep_horizon_days,
        )

    summary = {
        "forecast_date": (last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "total_predicted_demand": float(forecast_df["predicted_daily_demand"].sum()),
        "at_risk_items_count": int(len(at_risk)),
        "critical_items_count": int(len(at_risk[at_risk["risk_category"] == "critical"])) if not at_risk.empty else 0,
        "potential_waste_value": float(at_risk["estimated_revenue_loss"].sum()) if not at_risk.empty else 0.0,
        "potential_recovery_value": float(at_risk["potential_recovery"].sum()) if not at_risk.empty else 0.0,
        "promotions_created": len(promotions),
        "prep_ingredients_count": int(len(prep_plan)) if not prep_plan.empty else 0,
        "prep_total_cost": float(prep_plan["estimated_cost"].sum()) if not prep_plan.empty else 0.0,
    }

    report = {
        "demand_forecast": forecast_df,
        "inventory_prioritization": prioritized,
        "action_recommendations": recommendations,
        "at_risk_items": at_risk,
        "promotions": promotions,
        "prep_plan": prep_plan,
        "model_metrics": model_metrics,
        "summary": summary,
    }

    export_report(report, output_dir)
    return report


def export_report(report: Dict[str, object], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report["demand_forecast"].to_csv(Path(output_dir) / f"forecast_{timestamp}.csv", index=False)

    if isinstance(report["action_recommendations"], pd.DataFrame) and not report["action_recommendations"].empty:
        report["action_recommendations"].to_csv(
            Path(output_dir) / f"recommendations_{timestamp}.csv", index=False
        )

    if isinstance(report["prep_plan"], pd.DataFrame) and not report["prep_plan"].empty:
        report["prep_plan"].to_csv(Path(output_dir) / f"prep_plan_{timestamp}.csv", index=False)

    if report["promotions"]:
        pd.DataFrame(report["promotions"]).to_csv(
            Path(output_dir) / f"promotions_{timestamp}.csv", index=False
        )

    pd.DataFrame([report["summary"]]).to_csv(Path(output_dir) / f"summary_{timestamp}.csv", index=False)

    if report.get("model_metrics"):
        pd.DataFrame([report["model_metrics"]]).to_csv(
            Path(output_dir) / f"model_metrics_{timestamp}.csv", index=False
        )

    print(f"Reports exported to {output_dir}")
