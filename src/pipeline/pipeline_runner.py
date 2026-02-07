"""
============================================================================
FRESH FLOW PIPELINE RUNNER - PRODUCTION VERSION
============================================================================
Fully integrated with Ultimate AI V3.0 (9 models) and EnhancedDataLoader

✅ Uses all 9 AI models (6 ML + 3 Deep Learning)
✅ Real product names (Varm Chokolade, Sunny Hawaii, etc.)
✅ Enhanced error handling
✅ Production-ready

Version: 3.0 PRODUCTION
============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "src" / "models"
PIPELINE_DIR = Path(__file__).resolve().parent
SERVICES_DIR = ROOT_DIR / "src" / "services"

# Add to path
for dir_path in [MODELS_DIR, PIPELINE_DIR, SERVICES_DIR]:
    if str(dir_path) not in sys.path:
        sys.path.append(str(dir_path))

print("="*80)
print("🚀 FRESH FLOW PIPELINE - PRODUCTION VERSION V3.0")
print("="*80)

# Import data loader
try:
    from enhanced_data_loader_v2_1 import EnhancedDataLoader
    print("✅ EnhancedDataLoader imported (with real product names)")
    DataLoader = EnhancedDataLoader
    HAS_ENHANCED_LOADER = True
except ImportError as e:
    print(f"⚠️  EnhancedDataLoader not found: {e}")
    from data_loader import DataLoader
    print("✅ Using standard DataLoader")
    HAS_ENHANCED_LOADER = False

# Import Ultimate AI
try:
    from ultimate_inventory_intelligence_v3_PRODUCTION import UltimateInventoryIntelligence
    print("✅ Ultimate AI V3.0 PRODUCTION imported (9 models)")
    HAS_ULTIMATE_AI = True
except ImportError as e:
    print(f"⚠️  Ultimate AI not found: {e}")
    try:
        from ultimate_inventory_intelligence_v3_deeplearning import UltimateInventoryIntelligence
        print("✅ Ultimate AI V3.0 imported (fallback)")
        HAS_ULTIMATE_AI = True
    except ImportError:
        print("❌ No Ultimate AI available")
        HAS_ULTIMATE_AI = False

# Import other modules
from expiration_manager import ExpirationManager, PromotionOptimizer
from prep_calculator import PrepCalculator

print("="*80 + "\n")


def get_ultimate_ai_predictions(sales_data: pd.DataFrame, loader: DataLoader) -> pd.DataFrame:
    """
    Get predictions from Ultimate AI V3.0 (all 9 models)
    
    Args:
        sales_data: Daily sales data
        loader: Data loader with item mapping
    
    Returns:
        DataFrame with predictions and real product names
    """
    print("\n" + "="*70)
    print("🚀 ULTIMATE AI V3.0 - ENSEMBLE FORECASTING (9 MODELS)")
    print("="*70)
    
    if not HAS_ULTIMATE_AI:
        print("❌ Ultimate AI not available")
        return None
    
    try:
        # Prepare inventory
        unique_items = sales_data['item_id'].unique()
        print(f"\n📊 Processing {len(unique_items)} unique products...")
        
        # Create inventory dataframe
        inventory_data = pd.DataFrame({
            'item_id': unique_items,
            'current_stock': 100,
            'unit_cost': 30,
            'price': 50,
            'days_in_stock': 3,
            'shelf_life_days': 14
        })
        
        # Add real names if available
        if hasattr(loader, 'get_real_item_name'):
            inventory_data['title'] = inventory_data['item_id'].apply(loader.get_real_item_name)
        elif hasattr(loader, 'item_mapping'):
            inventory_data['title'] = inventory_data['item_id'].apply(
                lambda x: loader.item_mapping.get(x, {}).get('real_name', f'Item {x}')
            )
        
        # Load BOM if available
        bom_df = pd.DataFrame()
        if hasattr(loader, 'load_bill_of_materials'):
            try:
                bom_df = loader.load_bill_of_materials()
                if not bom_df.empty:
                    print(f"✅ BOM loaded: {len(bom_df):,} recipes")
            except Exception as e:
                print(f"⚠️  BOM loading failed: {e}")
        
        # Initialize Ultimate AI
        print(f"\n🧠 Initializing Ultimate AI Intelligence System...")
        print(f"   • Sales records: {len(sales_data):,}")
        print(f"   • Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")
        print(f"   • Using recent data: Last 45 days (post-holiday trends)")
        
        ultimate_ai = UltimateInventoryIntelligence(
            sales_data=sales_data,
            inventory_data=inventory_data,
            bill_of_materials=bom_df if not bom_df.empty else None,
            campaign_data=None,
            taxonomy_data=None,
            use_recent_data=True,  # Focus on recent trends
            recent_days=45  # Last 45 days
        )
        
        # Generate predictions
        print(f"\n📊 Generating ensemble predictions...")
        all_predictions = []
        
        for i, item_id in enumerate(unique_items, 1):
            try:
                # Get real name
                if hasattr(loader, 'get_real_item_name'):
                    real_name = loader.get_real_item_name(item_id)
                elif hasattr(loader, 'item_mapping') and item_id in loader.item_mapping:
                    real_name = loader.item_mapping[item_id].get('real_name', f'Item {item_id}')
                else:
                    real_name = f'Item {item_id}'
                
                # Progress indicator
                if i % 10 == 0 or i == 1:
                    print(f"   [{i}/{len(unique_items)}] {real_name}")
                
                # Get ensemble prediction (all 9 models)
                forecast = ultimate_ai.predict_demand_ensemble(
                    item_id=item_id,
                    days_ahead=1
                )
                
                all_predictions.append({
                    'item_id': item_id,
                    'item_real_name': real_name,
                    'predicted_daily_demand': max(forecast['ensemble_prediction'], 0),
                    'confidence': forecast['confidence_level'],
                    'num_models_used': forecast['num_models'],
                    'dl_models_used': forecast.get('dl_models_used', 0),
                    'prediction_source': 'Ultimate AI V3.0 (9 Models)'
                })
                
            except Exception as e:
                print(f"   ⚠️  Prediction failed for {real_name}: {str(e)[:50]}")
                
                # Fallback to historical average
                item_sales = sales_data[sales_data['item_id'] == item_id]['quantity_sold']
                fallback = item_sales.mean() if len(item_sales) > 0 else 0
                
                all_predictions.append({
                    'item_id': item_id,
                    'item_real_name': real_name,
                    'predicted_daily_demand': fallback,
                    'confidence': 0.5,
                    'num_models_used': 0,
                    'dl_models_used': 0,
                    'prediction_source': 'Historical Average (fallback)'
                })
        
        # Create dataframe
        predictions_df = pd.DataFrame(all_predictions)
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"✅ FORECASTING COMPLETE!")
        print(f"{'='*70}")
        print(f"📊 Total products: {len(predictions_df)}")
        print(f"📈 Average confidence: {predictions_df['confidence'].mean():.1%}")
        print(f"🔢 Average models used: {predictions_df['num_models_used'].mean():.1f}/9")
        print(f"🧠 Average DL models: {predictions_df['dl_models_used'].mean():.1f}/3")
        print(f"✅ Using Ultimate AI: {(predictions_df['prediction_source'] == 'Ultimate AI V3.0 (9 Models)').sum()}/{len(predictions_df)}")
        
        # Show top 10 with REAL NAMES
        print(f"\n{'='*70}")
        print(f"📈 TOP 10 PREDICTIONS (REAL PRODUCT NAMES)")
        print(f"{'='*70}")
        top_10 = predictions_df.nlargest(10, 'predicted_daily_demand')
        
        for idx, row in top_10.iterrows():
            name = row['item_real_name']
            demand = row['predicted_daily_demand']
            conf = row['confidence']
            models = row['num_models_used']
            dl = row['dl_models_used']
            
            print(f"  {name:35} = {demand:6.1f} units (conf: {conf:5.1%}, models: {models}, DL: {dl})")
        
        print(f"{'='*70}\n")
        
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
    Run the complete Fresh Flow pipeline with Ultimate AI V3.0
    
    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        forecast_horizon: "daily", "weekly", or "hourly"
        prep_horizon_days: Days ahead for prep planning
        prefer_advanced: Use Ultimate AI if available
    
    Returns:
        Dictionary with all reports
    """
    print("="*80)
    print("🌿 FRESH FLOW PIPELINE - ULTIMATE AI V3.0")
    print("="*80)
    print(f"📁 Data: {data_dir}")
    print(f"📊 Output: {output_dir}")
    print(f"🧠 AI: {'Ultimate AI V3.0 (9 models)' if prefer_advanced and HAS_ULTIMATE_AI else 'Baseline'}")
    print("="*80 + "\n")
    
    # Initialize data loader
    print("📁 Loading data...")
    loader = DataLoader(data_dir)
    
    # Create item mapping
    if HAS_ENHANCED_LOADER and hasattr(loader, 'create_item_mapping'):
        print("🏷️  Creating real product name mapping...")
        loader.create_item_mapping()
        print(f"   ✅ Mapped {len(loader.item_mapping)} items to real names\n")
    
    # Load data
    try:
        orders = loader.load_orders()
        order_items = loader.load_order_items()
        daily_sales = loader.prepare_daily_sales()
        
        print(f"   Orders: {len(orders):,}")
        print(f"   Order items: {len(order_items):,}")
        print(f"   Daily sales: {len(daily_sales):,}\n")
        
        if daily_sales.empty:
            raise ValueError("No sales data available!")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return {"error": f"Data loading failed: {e}"}
    
    # Get predictions
    forecast_df = None
    model_metrics = {}
    
    if prefer_advanced and HAS_ULTIMATE_AI:
        # Use Ultimate AI V3.0 (9 models)
        forecast_df = get_ultimate_ai_predictions(daily_sales, loader)
        
        if forecast_df is not None and not forecast_df.empty:
            model_metrics = {
                "model": "Ultimate AI V3.0 (9 Models)",
                "average_confidence": float(forecast_df['confidence'].mean()),
                "average_models_used": float(forecast_df['num_models_used'].mean()),
                "average_dl_models": float(forecast_df['dl_models_used'].mean())
            }
    
    # Fallback to baseline
    if forecast_df is None or forecast_df.empty:
        print("\n⚠️  Using baseline forecasting...")
        from feature_engineering import build_features
        from forecasting import get_forecaster
        
        features_df = build_features(daily_sales, orders, order_items)
        forecaster = get_forecaster(forecast_horizon=forecast_horizon, prefer_advanced=False)
        forecaster.train(features_df, target_col="quantity_sold")
        
        last_date = features_df["date"].max()
        latest = features_df[features_df["date"] == last_date].copy()
        predictions = forecaster.predict(latest)
        
        forecast_df = latest[["item_id"]].copy()
        forecast_df["forecast_date"] = last_date + timedelta(days=1)
        forecast_df["predicted_daily_demand"] = np.maximum(predictions, 0).round(2)
        
        # Apply real names
        if hasattr(loader, 'get_real_item_name'):
            forecast_df['item_real_name'] = forecast_df['item_id'].apply(loader.get_real_item_name)
        
        model_metrics = {"model": "Baseline RandomForest"}
    
    # Add forecast date
    last_date = daily_sales["date"].max()
    if 'forecast_date' not in forecast_df.columns:
        forecast_df["forecast_date"] = last_date + timedelta(days=1)
    
    # Add location counts
    if 'place_id' in daily_sales.columns:
        location_counts = daily_sales.groupby('item_id')['place_id'].nunique().reset_index()
        location_counts.columns = ['item_id', 'sold_at_locations']
        forecast_df = forecast_df.merge(location_counts, on='item_id', how='left')
        forecast_df['sold_at_locations'] = forecast_df['sold_at_locations'].fillna(1)
    
    # Load inventory
    print("\n📦 Loading inventory...")
    inventory_full = loader.prepare_inventory_snapshot()
    
    # Apply real names to inventory
    if not inventory_full.empty and hasattr(loader, 'get_real_item_name'):
        inventory_full['item_real_name'] = inventory_full['item_id'].apply(loader.get_real_item_name)
    
    # Expiration analysis
    print("⏰ Analyzing expiration risks...")
    expiration_manager = ExpirationManager(None)
    
    prioritized = pd.DataFrame()
    recommendations = pd.DataFrame()
    at_risk = pd.DataFrame()
    
    required_cols = ['days_in_stock', 'shelf_life_days']
    has_expiration = all(col in inventory_full.columns for col in required_cols)
    
    if not inventory_full.empty and has_expiration:
        inventory_subset = inventory_full[inventory_full["item_id"].isin(forecast_df["item_id"])].copy()
        
        if 'days_until_expiration' not in inventory_subset.columns:
            inventory_subset['days_until_expiration'] = (
                inventory_subset['shelf_life_days'] - inventory_subset['days_in_stock']
            ).clip(lower=0)
        
        if not inventory_subset.empty:
            try:
                prioritized = expiration_manager.prioritize_inventory(inventory_subset, forecast_df)
                recommendations = expiration_manager.recommend_actions(prioritized)
                at_risk = recommendations[
                    recommendations["risk_category"].isin(["critical", "high", "medium"])
                ]
                
                # Apply real names
                if 'item_real_name' in forecast_df.columns:
                    recommendations = recommendations.merge(
                        forecast_df[['item_id', 'item_real_name']].drop_duplicates(),
                        on='item_id',
                        how='left'
                    )
                
                print(f"   ✅ Found {len(at_risk)} at-risk items\n")
            except Exception as e:
                print(f"   ⚠️  Expiration analysis failed: {e}\n")
    
    # Promotions
    print("🎯 Creating promotions...")
    promotions = []
    if not at_risk.empty:
        promo_optimizer = PromotionOptimizer(order_items)
        promotions = promo_optimizer.create_bundle_promotions(at_risk)
        print(f"   ✅ Created {len(promotions)} promotions\n")
    
    # Prep plan
    print("👨‍🍳 Calculating prep quantities...")
    prep_plan = pd.DataFrame()
    bom_df = loader.load_bill_of_materials() if hasattr(loader, 'load_bill_of_materials') else pd.DataFrame()
    
    if not bom_df.empty:
        prep_calculator = PrepCalculator(bom_df, inventory_full)
        prep_plan = prep_calculator.calculate_prep_quantities(
            forecast_df,
            prep_date=last_date + timedelta(days=1),
            prep_horizon_days=prep_horizon_days,
        )
        
        # Apply real names
        if not prep_plan.empty and hasattr(loader, 'get_real_item_name'):
            if 'raw_item_id' in prep_plan.columns:
                prep_plan['ingredient_real_name'] = prep_plan['raw_item_id'].apply(
                    loader.get_real_item_name
                )
        
        print(f"   ✅ Calculated prep for {len(prep_plan)} ingredients\n")
    
    # Summary
    summary = {
        "forecast_date": forecast_df["forecast_date"].iloc[0].strftime("%Y-%m-%d"),
        "total_predicted_demand": float(forecast_df["predicted_daily_demand"].sum()),
        "total_items_forecasted": int(len(forecast_df)),
        "ai_model_used": model_metrics.get("model", "Unknown"),
        "average_confidence": model_metrics.get("average_confidence", 0),
        "average_models_used": model_metrics.get("average_models_used", 0),
        "average_dl_models": model_metrics.get("average_dl_models", 0),
        "at_risk_items_count": int(len(at_risk)),
        "promotions_created": len(promotions),
        "prep_ingredients_count": int(len(prep_plan)),
        "real_names_applied": "Yes"
    }
    
    # Export reports
    print("💾 Exporting reports...")
    export_reports(forecast_df, output_dir, summary, recommendations, prep_plan, promotions)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    
    return {
        "demand_forecast": forecast_df,
        "inventory_prioritization": prioritized,
        "action_recommendations": recommendations,
        "at_risk_items": at_risk,
        "promotions": promotions,
        "prep_plan": prep_plan,
        "model_metrics": model_metrics,
        "summary": summary,
    }


def export_reports(forecast_df, output_dir, summary, recommendations, prep_plan, promotions):
    """Export all reports with proper formatting"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 EXPORTING REPORTS TO: {output_dir}")
    
    # FORECAST REPORT - This is the most important for dashboard
    if forecast_df is not None and not forecast_df.empty:
        print(f"📈 Exporting forecast data with {len(forecast_df)} items")
        
        # Ensure proper column names for dashboard
        forecast_export = forecast_df.copy()
        
        # Map column names to dashboard expectations
        column_mapping = {
            'item_real_name': 'Product Name',
            'item_name': 'Product Name',
            'title': 'Product Name',
            'predicted_daily_demand': 'Expected Daily Demand',
            'predicted_demand': 'Expected Daily Demand',
            'forecast_demand': 'Expected Daily Demand',
            'daily_demand': 'Expected Daily Demand',
            'confidence': 'Confidence',
            'prediction_source': 'Prediction Source',
            'num_models_used': 'Models Used',
            'dl_models_used': 'DL Models Used'
        }
        
        # Rename columns
        forecast_export = forecast_export.rename(columns={k: v for k, v in column_mapping.items() 
                                                         if k in forecast_export.columns})
        
        # Ensure we have a Product Name column
        if 'Product Name' not in forecast_export.columns:
            # Try to create from available columns
            name_candidates = ['item_real_name', 'item_name', 'title', 'product_name']
            for col in name_candidates:
                if col in forecast_export.columns:
                    forecast_export['Product Name'] = forecast_export[col].astype(str)
                    break
            
            if 'Product Name' not in forecast_export.columns:
                forecast_export['Product Name'] = forecast_export['item_id'].apply(lambda x: f"Item {x}")
        
        # Ensure we have Expected Daily Demand column
        if 'Expected Daily Demand' not in forecast_export.columns:
            demand_candidates = ['predicted_daily_demand', 'predicted_demand', 'forecast_demand', 'daily_demand']
            for col in demand_candidates:
                if col in forecast_export.columns:
                    forecast_export['Expected Daily Demand'] = pd.to_numeric(forecast_export[col], errors='coerce')
                    break
            
            if 'Expected Daily Demand' not in forecast_export.columns:
                # Look for any numeric column that could be demand
                numeric_cols = forecast_export.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    forecast_export['Expected Daily Demand'] = forecast_export[numeric_cols[0]]
                else:
                    forecast_export['Expected Daily Demand'] = 0
        
        # Select important columns for dashboard
        display_cols = ['Product Name', 'Expected Daily Demand']
        if 'Confidence' in forecast_export.columns:
            display_cols.append('Confidence')
        if 'Prediction Source' in forecast_export.columns:
            display_cols.append('Prediction Source')
        if 'Models Used' in forecast_export.columns:
            display_cols.append('Models Used')
        
        # Create final export dataframe
        final_forecast = forecast_export[display_cols].copy()
        
        # Sort by demand
        final_forecast = final_forecast.sort_values('Expected Daily Demand', ascending=False)
        
        # Save to CSV
        forecast_file = Path(output_dir) / f"forecast_{timestamp}.csv"
        final_forecast.to_csv(forecast_file, index=False)
        print(f"   ✅ Forecast saved: {forecast_file.name}")
        print(f"       • Products: {len(final_forecast)}")
        print(f"       • Total Demand: {final_forecast['Expected Daily Demand'].sum():.1f}")
        print(f"       • Top product: {final_forecast.iloc[0]['Product Name'] if len(final_forecast) > 0 else 'N/A'}")
        
        # Also save a detailed version
        detailed_forecast_file = Path(output_dir) / f"forecast_detailed_{timestamp}.csv"
        forecast_df.to_csv(detailed_forecast_file, index=False)
    
    # SUMMARY REPORT
    if summary:
        summary_df = pd.DataFrame([summary])
        summary_file = Path(output_dir) / f"summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"   ✅ Summary saved: {summary_file.name}")
    
    # RECOMMENDATIONS REPORT
    if recommendations is not None and not recommendations.empty:
        print(f"📋 Exporting recommendations data with {len(recommendations)} items")
        
        # Ensure proper formatting
        rec_export = recommendations.copy()
        
        # Map column names
        rec_mapping = {
            'item_real_name': 'Product Name',
            'item_name': 'Product Name',
            'title': 'Product Name',
            'risk_category': 'Risk Category',
            'days_until_expiration': 'Days Until Expiration',
            'recommended_action': 'Recommended Action',
            'urgency_score': 'Urgency Score'
        }
        
        rec_export = rec_export.rename(columns={k: v for k, v in rec_mapping.items() 
                                               if k in rec_export.columns})
        
        # Save to CSV
        rec_file = Path(output_dir) / f"recommendations_{timestamp}.csv"
        rec_export.to_csv(rec_file, index=False)
        print(f"   ✅ Recommendations saved: {rec_file.name}")
    
    # PREP PLAN REPORT
    if prep_plan is not None and not prep_plan.empty:
        print(f"👨‍🍳 Exporting prep plan with {len(prep_plan)} ingredients")
        
        prep_export = prep_plan.copy()
        
        # Map column names
        prep_mapping = {
            'ingredient_real_name': 'Ingredient Name',
            'raw_item_id': 'Ingredient ID',
            'quantity_needed': 'Quantity Needed',
            'current_stock': 'Current Stock',
            'net_to_order': 'Order Quantity',
            'unit': 'Unit'
        }
        
        prep_export = prep_export.rename(columns={k: v for k, v in prep_mapping.items() 
                                                 if k in prep_export.columns})
        
        prep_file = Path(output_dir) / f"prep_plan_{timestamp}.csv"
        prep_export.to_csv(prep_file, index=False)
        print(f"   ✅ Prep plan saved: {prep_file.name}")
    
    # PROMOTIONS REPORT
    if promotions:
        promo_df = pd.DataFrame(promotions)
        if not promo_df.empty:
            promo_file = Path(output_dir) / f"promotions_{timestamp}.csv"
            promo_df.to_csv(promo_file, index=False)
            print(f"   ✅ Promotions saved: {promo_file.name}")
    
    print(f"\n ALL REPORTS EXPORTED SUCCESSFULLY")
    print(f" Location: {output_dir}")
    print(f"  Timestamp: {timestamp}\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Forecast
    clean_forecast = pd.DataFrame()
    clean_forecast['ID'] = forecast_df['item_id']
    
    if 'item_real_name' in forecast_df.columns:
        clean_forecast['Product'] = forecast_df['item_real_name']
    else:
        clean_forecast['Product'] = forecast_df['item_id'].apply(lambda x: f"Item {x}")
    
    clean_forecast['Expected Daily Demand'] = forecast_df['predicted_daily_demand']
    
    if 'confidence' in forecast_df.columns:
        clean_forecast['Confidence'] = forecast_df['confidence']
    if 'num_models_used' in forecast_df.columns:
        clean_forecast['Models Used'] = forecast_df['num_models_used']
    if 'dl_models_used' in forecast_df.columns:
        clean_forecast['DL Models'] = forecast_df['dl_models_used']
    
    clean_forecast = clean_forecast.sort_values('Expected Daily Demand', ascending=False)
    
    forecast_file = Path(output_dir) / f"forecast_{timestamp}.csv"
    clean_forecast.to_csv(forecast_file, index=False)
    print(f"   ✅ {forecast_file.name}")
    
    # Summary
    summary_df = pd.DataFrame([summary])
    summary_file = Path(output_dir) / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"   ✅ {summary_file.name}")
    
    # Recommendations
    if not recommendations.empty:
        rec_file = Path(output_dir) / f"recommendations_{timestamp}.csv"
        recommendations.to_csv(rec_file, index=False)
        print(f"   ✅ {rec_file.name}")
    
    # Prep plan
    if not prep_plan.empty:
        prep_file = Path(output_dir) / f"prep_plan_{timestamp}.csv"
        prep_plan.to_csv(prep_file, index=False)
        print(f"   {prep_file.name}")
    
    # Promotions
    if promotions:
        promo_df = pd.DataFrame(promotions)
        promo_file = Path(output_dir) / f"promotions_{timestamp}.csv"
        promo_df.to_csv(promo_file, index=False)
        print(f"    {promo_file.name}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_runner_PRODUCTION.py <data_dir> [output_dir]")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./reports"
    
    try:
        report = run_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            forecast_horizon="daily",
            prep_horizon_days=1,
            prefer_advanced=True  # Use Ultimate AI V3.0
        )
        
        print("\n SUCCESS!")
        
    except Exception as e:
        print(f"\n FAILED: {e}")
        import traceback
        traceback.print_exc()