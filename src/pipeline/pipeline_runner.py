"""
============================================================================
FRESH FLOW PIPELINE RUNNER - ULTIMATE AI V3.0 WITH REAL NAMES
============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import pandas as pd

# Ensure models directory is importable.
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "src" / "models"
PIPELINE_DIR = Path(__file__).resolve().parent
SERVICES_DIR = ROOT_DIR / "src" / "services"

# Add paths for imports
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))
if str(PIPELINE_DIR) not in sys.path:
    sys.path.append(str(PIPELINE_DIR))
if str(SERVICES_DIR) not in sys.path:
    sys.path.append(str(SERVICES_DIR))

print("Python path includes:")
print(f"  Models: {MODELS_DIR}")
print(f"  Services: {SERVICES_DIR}")

# Import the enhanced data loader
try:
    from enhanced_data_loader_v2_1 import EnhancedDataLoader
    print("✅ Using EnhancedDataLoader with built-in name mapping")
    DataLoader = EnhancedDataLoader
    HAS_ENHANCED_LOADER = True
except ImportError as e:
    print(f"⚠️  Could not import EnhancedDataLoader: {e}")
    print("Falling back to standard DataLoader")
    from data_loader import DataLoader
    HAS_ENHANCED_LOADER = False

from expiration_manager import ExpirationManager, PromotionOptimizer
from prep_calculator import PrepCalculator

# Try to import Ultimate AI
try:
    from ultimate_ai_solution_v3 import UltimateInventoryIntelligence
    print("✅ Ultimate AI Solution v3.0 imported successfully")
    HAS_ULTIMATE_AI = True
except ImportError as e:
    print(f"⚠️  Could not import Ultimate AI: {e}")
    HAS_ULTIMATE_AI = False


def get_ultimate_ai_predictions(sales_data: pd.DataFrame, loader: EnhancedDataLoader):
    """
    Get predictions from Ultimate AI v3.0 using actual data
    
    Args:
        sales_data: Daily sales data with item_id and quantity_sold
        loader: Data loader instance with item mapping
    
    Returns:
        DataFrame with predictions and real product names
    """
    print("\n" + "="*70)
    print("🚀 USING ULTIMATE AI SOLUTION V3.0 FOR PREDICTIONS")
    print("="*70)
    
    if not HAS_ULTIMATE_AI:
        print("❌ Ultimate AI not available, falling back to historical average")
        return None
    
    try:
        # Prepare inventory data
        unique_items = sales_data['item_id'].unique()
        inventory_data = pd.DataFrame({
            'item_id': unique_items,
            'current_stock': 100,
            'unit_cost': 30,
            'price': 50
        })
        
        # Load BOM if available
        bom_df = loader.load_bill_of_materials() if hasattr(loader, 'load_bill_of_materials') else pd.DataFrame()
        
        # Initialize Ultimate AI
        print("Initializing Ultimate AI Intelligence System...")
        print(f"  Sales records: {len(sales_data):,}")
        print(f"  Unique items: {len(unique_items)}")
        print(f"  Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")
        
        ultimate_ai = UltimateInventoryIntelligence(
            sales_data=sales_data,
            inventory_data=inventory_data,
            bill_of_materials=bom_df if not bom_df.empty else None,
            campaign_data=None,
            taxonomy_data=None
        )
        
        # Get predictions for each item
        print(f"\n📊 Generating predictions for {len(unique_items)} items...")
        all_predictions = []
        
        for i, item_id in enumerate(unique_items, 1):
            try:
                # Get real name from loader's mapping
                if hasattr(loader, 'get_real_item_name'):
                    real_name = loader.get_real_item_name(item_id)
                elif hasattr(loader, 'item_mapping') and item_id in loader.item_mapping:
                    real_name = loader.item_mapping[item_id].get('real_name', f'Item {item_id}')
                else:
                    real_name = f'Item {item_id}'
                
                if i % 5 == 0 or i == 1:
                    print(f"  [{i}/{len(unique_items)}] Predicting: {real_name} (ID: {item_id})")
                
                # Get ensemble prediction from Ultimate AI
                forecast_result = ultimate_ai.predict_demand_ensemble(
                    item_id=item_id,
                    days_ahead=1
                )
                
                pred = forecast_result['ensemble_prediction']
                confidence = forecast_result['confidence_level']
                models_used = forecast_result['num_models']
                
                all_predictions.append({
                    'item_id': item_id,
                    'item_real_name': real_name,
                    'predicted_daily_demand': max(pred, 0),
                    'confidence': confidence,
                    'num_models_used': models_used,
                    'prediction_source': 'Ultimate AI v3.0'
                })
                
            except Exception as e:
                print(f"  ⚠️  Prediction failed for item {item_id}: {str(e)[:50]}")
                # Fallback to historical average
                item_sales = sales_data[sales_data['item_id'] == item_id]['quantity_sold']
                fallback_pred = item_sales.mean() if len(item_sales) > 0 else 0
                
                all_predictions.append({
                    'item_id': item_id,
                    'item_real_name': real_name,
                    'predicted_daily_demand': fallback_pred,
                    'confidence': 0.5,
                    'num_models_used': 0,
                    'prediction_source': 'Historical Average (fallback)'
                })
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame(all_predictions)
        
        print(f"\n✅ PREDICTIONS COMPLETE!")
        print(f"  Total items: {len(predictions_df)}")
        print(f"  Average confidence: {predictions_df['confidence'].mean():.1%}")
        print(f"  Average models used: {predictions_df['num_models_used'].mean():.1f}/9")
        print(f"  Using Ultimate AI: {(predictions_df['prediction_source'] == 'Ultimate AI v3.0').sum()}/{len(predictions_df)}")
        
        # Show top predictions with REAL NAMES
        print(f"\n📈 TOP 10 PREDICTIONS (REAL PRODUCT NAMES):")
        top_preds = predictions_df.sort_values('predicted_daily_demand', ascending=False).head(10)
        for idx, row in top_preds.iterrows():
            name = row['item_real_name']
            demand = row['predicted_daily_demand']
            conf = row['confidence']
            models = row['num_models_used']
            print(f"  {name:30} = {demand:6.1f} units (conf: {conf:.1%}, models: {models})")
        
        return predictions_df
        
    except Exception as e:
        print(f"❌ Ultimate AI failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_pipeline(
    data_dir: str,
    output_dir: str,
    forecast_horizon: str = "daily",
    prep_horizon_days: int = 1,
    prefer_advanced: bool = True,
) -> Dict:
    """
    Run the complete Fresh Flow pipeline
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory for reports
        forecast_horizon: Forecasting timeframe ("daily", "weekly", "hourly")
        prep_horizon_days: Days ahead for prep planning
        prefer_advanced: Use Ultimate AI if available
    
    Returns:
        Dictionary with all reports and predictions
    """
    print("=" * 70)
    print("🌿 FRESH FLOW PIPELINE - ULTIMATE AI V3.0")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Advanced AI: {prefer_advanced and HAS_ULTIMATE_AI}")
    print("=" * 70)
    
    # Initialize data loader
    print("\n📁 Initializing Data Loader...")
    loader = DataLoader(data_dir)
    
    # Create item name mapping if using enhanced loader
    if HAS_ENHANCED_LOADER and hasattr(loader, 'create_item_mapping'):
        print("🏷️  Creating item name mapping...")
        loader.create_item_mapping()
        print(f"  ✅ Mapped {len(loader.item_mapping)} items to real names")
    
    # Load data
    print("\n📊 Loading Data...")
    try:
        orders = loader.load_orders()
        order_items = loader.load_order_items()
        daily_sales = loader.prepare_daily_sales()
        
        print(f"  Orders: {len(orders):,}")
        print(f"  Order items: {len(order_items):,}")
        print(f"  Daily sales: {len(daily_sales):,}")
        
        if daily_sales.empty:
            raise ValueError("No sales data available!")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return {"error": f"Data loading failed: {e}"}
    
    # Get predictions
    forecast_df = None
    model_metrics = {}
    
    if prefer_advanced and HAS_ULTIMATE_AI:
        # Use Ultimate AI
        forecast_df = get_ultimate_ai_predictions(daily_sales, loader)
        
        if forecast_df is not None and not forecast_df.empty:
            model_metrics = {
                "model": "Ultimate AI v3.0",
                "average_confidence": float(forecast_df['confidence'].mean()),
                "average_models_used": float(forecast_df['num_models_used'].mean())
            }
    
    # Fallback to baseline if Ultimate AI failed
    if forecast_df is None or forecast_df.empty:
        print("\n⚠️  Using baseline forecasting...")
        from feature_engineering import build_features
        from forecasting import get_forecaster
        
        features_df = build_features(daily_sales, orders, order_items)
        
        forecaster = get_forecaster(forecast_horizon=forecast_horizon, prefer_advanced=False)
        forecaster.train(features_df, target_col="quantity_sold")
        
        last_date = features_df["date"].max()
        latest_rows = features_df[features_df["date"] == last_date].copy()
        predictions = forecaster.predict(latest_rows)
        
        forecast_df = latest_rows[["item_id"]].copy()
        forecast_df["forecast_date"] = last_date + timedelta(days=1)
        forecast_df["predicted_daily_demand"] = np.maximum(predictions, 0).round(2)
        
        # Apply real names from loader
        if hasattr(loader, 'get_real_item_name'):
            forecast_df['item_real_name'] = forecast_df['item_id'].apply(loader.get_real_item_name)
        elif hasattr(loader, 'item_mapping'):
            forecast_df['item_real_name'] = forecast_df['item_id'].apply(
                lambda x: loader.item_mapping.get(x, {}).get('real_name', f'Item {x}')
            )
        else:
            forecast_df['item_real_name'] = forecast_df['item_id'].apply(lambda x: f'Item {x}')
        
        model_metrics = {"model": "Baseline RandomForest"}
    
    # Add forecast date if missing
    last_date = daily_sales["date"].max() if 'date' in daily_sales.columns else datetime.now()
    if 'forecast_date' not in forecast_df.columns:
        forecast_df["forecast_date"] = last_date + timedelta(days=1)
    
    # Add location counts
    if 'place_id' in daily_sales.columns:
        location_counts = daily_sales.groupby('item_id')['place_id'].nunique().reset_index()
        location_counts.columns = ['item_id', 'sold_at_locations']
        forecast_df = forecast_df.merge(location_counts, on='item_id', how='left')
        forecast_df['sold_at_locations'].fillna(1, inplace=True)
    else:
        forecast_df['sold_at_locations'] = 1
    
    # Load inventory
    print("\n📦 Loading inventory data...")
    inventory_full = loader.prepare_inventory_snapshot()
    
    # Apply real names to inventory
    if not inventory_full.empty and 'item_id' in inventory_full.columns:
        if hasattr(loader, 'get_real_item_name'):
            inventory_full['item_real_name'] = inventory_full['item_id'].apply(loader.get_real_item_name)
        elif hasattr(loader, 'item_mapping'):
            inventory_full['item_real_name'] = inventory_full['item_id'].apply(
                lambda x: loader.item_mapping.get(x, {}).get('real_name', f'Item {x}')
            )
    
    # Expiration analysis
    expiration_manager = ExpirationManager(None)
    
    # Check if inventory has required columns for expiration analysis
    required_expiration_cols = ['days_in_stock', 'shelf_life_days']
    has_expiration_data = all(col in inventory_full.columns for col in required_expiration_cols)
    
    if not inventory_full.empty and has_expiration_data:
        inventory_for_expiration = inventory_full[
            inventory_full["item_id"].isin(forecast_df["item_id"])
        ].copy()
        
        # Calculate days_until_expiration if not present
        if 'days_until_expiration' not in inventory_for_expiration.columns:
            inventory_for_expiration['days_until_expiration'] = (
                inventory_for_expiration['shelf_life_days'] - 
                inventory_for_expiration['days_in_stock']
            ).clip(lower=0)
        
        if not inventory_for_expiration.empty:
            print("⏰ Analyzing inventory expiration...")
            try:
                prioritized = expiration_manager.prioritize_inventory(inventory_for_expiration, forecast_df)
                recommendations = expiration_manager.recommend_actions(prioritized)
                at_risk = recommendations[recommendations["risk_category"].isin(["critical", "high", "medium"])]
                print(f"  Found {len(at_risk)} at-risk items")
                
                # Apply real names to recommendations
                if 'item_real_name' in forecast_df.columns:
                    recommendations = recommendations.merge(
                        forecast_df[['item_id', 'item_real_name']].drop_duplicates(),
                        on='item_id',
                        how='left'
                    )
            except Exception as e:
                print(f"  ⚠️  Expiration analysis failed: {e}")
                prioritized = pd.DataFrame()
                recommendations = pd.DataFrame()
                at_risk = pd.DataFrame()
        else:
            prioritized = pd.DataFrame()
            recommendations = pd.DataFrame()
            at_risk = pd.DataFrame()
    else:
        if inventory_full.empty:
            print("⚠️  No inventory data available")
        else:
            print(f"⚠️  Inventory missing expiration columns: {[col for col in required_expiration_cols if col not in inventory_full.columns]}")
        prioritized = pd.DataFrame()
        recommendations = pd.DataFrame()
        at_risk = pd.DataFrame()
    
    # Promotions
    promotions = []
    if not at_risk.empty:
        print("🎯 Creating promotions...")
        promo_optimizer = PromotionOptimizer(order_items)
        promotions = promo_optimizer.create_bundle_promotions(at_risk)
        print(f"  Created {len(promotions)} promotions")
    
    # Prep plan
    prep_plan = pd.DataFrame()
    bom_df = loader.load_bill_of_materials() if hasattr(loader, 'load_bill_of_materials') else pd.DataFrame()
    
    if not bom_df.empty:
        print("👨‍🍳 Calculating prep quantities...")
        prep_calculator = PrepCalculator(bom_df, inventory_full)
        prep_plan = prep_calculator.calculate_prep_quantities(
            forecast_df,
            prep_date=last_date + timedelta(days=1),
            prep_horizon_days=prep_horizon_days,
        )
        
        # Apply real names to prep plan
        if not prep_plan.empty and hasattr(loader, 'get_real_item_name'):
            if 'raw_item_id' in prep_plan.columns:
                prep_plan['ingredient_real_name'] = prep_plan['raw_item_id'].apply(
                    loader.get_real_item_name
                )
    
    # Create summary
    summary = {
        "forecast_date": forecast_df["forecast_date"].iloc[0].strftime("%Y-%m-%d") if not forecast_df.empty else "N/A",
        "total_predicted_demand": float(forecast_df["predicted_daily_demand"].sum()),
        "total_items_forecasted": int(len(forecast_df)),
        "ai_model_used": model_metrics.get("model", "Unknown"),
        "at_risk_items_count": int(len(at_risk)),
        "promotions_created": len(promotions),
        "prep_ingredients_count": int(len(prep_plan)),
        "real_names_applied": "Yes" if 'item_real_name' in forecast_df.columns else "No"
    }
    
    # Export reports
    print("\n💾 Exporting reports...")
    export_reports(forecast_df, output_dir, summary, recommendations, prep_plan, promotions)
    
    # Create report
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
    
    print("\n✅ PIPELINE COMPLETE!")
    print("=" * 70)
    
    return report


def export_reports(forecast_df: pd.DataFrame, output_dir: str, summary: Dict, 
                  recommendations: pd.DataFrame, prep_plan: pd.DataFrame, 
                  promotions: list):
    """Export all reports to output directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === FORECAST REPORT ===
    clean_forecast = pd.DataFrame()
    clean_forecast['ID'] = forecast_df['item_id']
    
    if 'item_real_name' in forecast_df.columns:
        clean_forecast['Product'] = forecast_df['item_real_name']
    else:
        clean_forecast['Product'] = forecast_df['item_id'].apply(lambda x: f"Item {x}")
    
    clean_forecast['Expected Demand'] = forecast_df['predicted_daily_demand']
    
    if 'confidence' in forecast_df.columns:
        clean_forecast['Confidence'] = forecast_df['confidence']
    if 'num_models_used' in forecast_df.columns:
        clean_forecast['Models Used'] = forecast_df['num_models_used']
    if 'sold_at_locations' in forecast_df.columns:
        clean_forecast['Locations'] = forecast_df['sold_at_locations']
    if 'forecast_date' in forecast_df.columns:
        clean_forecast['Forecast Date'] = forecast_df['forecast_date']
    
    clean_forecast = clean_forecast.sort_values('Expected Demand', ascending=False)
    
    forecast_file = Path(output_dir) / f"forecast_{timestamp}.csv"
    clean_forecast.to_csv(forecast_file, index=False)
    
    print(f"  ✅ Forecast: {forecast_file.name}")
    
    # === SUMMARY REPORT ===
    summary_df = pd.DataFrame([summary])
    summary_file = Path(output_dir) / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✅ Summary: {summary_file.name}")
    
    # === RECOMMENDATIONS REPORT ===
    if not recommendations.empty:
        rec_file = Path(output_dir) / f"recommendations_{timestamp}.csv"
        recommendations.to_csv(rec_file, index=False)
        print(f"  ✅ Recommendations: {rec_file.name}")
    
    # === PREP PLAN REPORT ===
    if not prep_plan.empty:
        prep_file = Path(output_dir) / f"prep_plan_{timestamp}.csv"
        prep_plan.to_csv(prep_file, index=False)
        print(f"  ✅ Prep Plan: {prep_file.name}")
    
    # === PROMOTIONS REPORT ===
    if promotions:
        promo_df = pd.DataFrame(promotions)
        promo_file = Path(output_dir) / f"promotions_{timestamp}.csv"
        promo_df.to_csv(promo_file, index=False)
        print(f"  ✅ Promotions: {promo_file.name}")
    
    # Show sample of results
    print(f"\n📊 TOP 5 PREDICTIONS:")
    for idx, row in clean_forecast.head(5).iterrows():
        product = row['Product']
        demand = row['Expected Demand']
        print(f"  {idx+1}. {product:30} = {demand:6.1f} units")


if __name__ == "__main__":
    import sys
    
    print("🌿 FRESH FLOW PIPELINE")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_runner.py <data_dir> <output_dir>")
        print("Example: python pipeline_runner.py ./data ./output")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    try:
        report = run_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            forecast_horizon="daily",
            prep_horizon_days=1,
            prefer_advanced=True  # Use Ultimate AI
        )
        
        print("\n✅ PIPELINE EXECUTION COMPLETE")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
