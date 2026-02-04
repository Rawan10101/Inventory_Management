"""
Test script for Advanced ML Inventory Service
"""

from src.models.data_loader import DataLoader
from src.services.inventory_service_ml import AdvancedInventoryService
import pandas as pd

# Load data
print("Loading data...")
loader = DataLoader(r"D:\Inventory_Management\data\Inventory_Management")
daily_sales = loader.prepare_daily_sales()
inventory_snapshot = loader.prepare_inventory_snapshot()

# Initialize ML service
print("\nInitializing ML Service...")
service = AdvancedInventoryService(inventory_data=inventory_snapshot, sales_data=daily_sales)

# Pick sample items (preferably with enough history)
sample_items = daily_sales.groupby('item_id').size().sort_values(ascending=False).head(3).index.tolist()

print(f"\n{'='*70}")
print(f"Testing {len(sample_items)} items with ML models")
print(f"{'='*70}\n")

for i, item_id in enumerate(sample_items, 1):
    print(f"\n{'='*70}")
    print(f"Item {i}/{len(sample_items)}: {item_id}")
    print(f"{'='*70}")
    
    try:
        # Get item info
        item_info = daily_sales[daily_sales['item_id'] == item_id].iloc[0]
        print(f"📦 Item: {item_info.get('title', 'Unknown')}")
        print(f"🏪 Merchant: {item_info.get('title_place', 'Unknown')}")
        
        # Test different models
        print(f"\n📊 Model Predictions:")
        print(f"{'-'*70}")
        
        # ARIMA
        arima_result = service.predict_arima(item_id)
        if 'prediction' in arima_result and arima_result['prediction']:
            print(f"  ARIMA:     {arima_result['prediction']:.2f} units/day")
        else:
            print(f"  ARIMA:     ❌ {arima_result.get('error', 'Failed')}")
        
        # Prophet
        prophet_result = service.predict_prophet(item_id)
        if 'prediction' in prophet_result and prophet_result['prediction']:
            print(f"  Prophet:   {prophet_result['prediction']:.2f} units/day")
        else:
            print(f"  Prophet:   ❌ {prophet_result.get('error', 'Failed')}")
        
        # XGBoost
        xgb_result = service.predict_xgboost(item_id)
        if 'prediction' in xgb_result and xgb_result['prediction']:
            print(f"  XGBoost:   {xgb_result['prediction']:.2f} units/day")
            if 'r2_score' in xgb_result:
                print(f"             (R² Score: {xgb_result['r2_score']:.3f})")
        else:
            print(f"  XGBoost:   ❌ {xgb_result.get('error', 'Failed')}")
        
        # LSTM
        lstm_result = service.predict_lstm(item_id)
        if 'prediction' in lstm_result and lstm_result['prediction']:
            print(f"  LSTM:      {lstm_result['prediction']:.2f} units/day")
        else:
            print(f"  LSTM:      ❌ {lstm_result.get('error', 'Failed')}")
        
        # Ensemble
        ensemble_result = service.predict_ensemble(item_id)
        if 'prediction' in ensemble_result and ensemble_result['prediction']:
            print(f"\n  🎯 ENSEMBLE: {ensemble_result['prediction']:.2f} units/day")
            print(f"     Models used: {', '.join(ensemble_result.get('models_used', []))}")
            print(f"     Confidence: {ensemble_result.get('confidence', 0)*100:.0f}%")
        
        # Comprehensive recommendations
        print(f"\n📋 Comprehensive Analysis:")
        print(f"{'-'*70}")
        
        recommendations = service.generate_comprehensive_recommendations(item_id)
        
        print(f"  Predicted Demand:    {recommendations['predictions']['ensemble']} units/day")
        print(f"  Reorder Point:       {recommendations['inventory_metrics']['reorder_point']} units")
        print(f"  Current Stock:       {recommendations['inventory_metrics']['current_stock']} units")
        print(f"  Stock Risk Level:    {recommendations['inventory_metrics']['stock_risk']}")
        print(f"  Recent Trend:        {recommendations['trends']['trend_direction']} ({recommendations['trends']['trend_percentage']:+.1f}%)")
        print(f"  Confidence Score:    {recommendations['confidence_score']*100:.0f}%")
        print(f"\n  💡 Recommendation:   {recommendations['action']}")
        
    except Exception as e:
        print(f"❌ Error processing item {item_id}: {e}")

print(f"\n{'='*70}")
print(f"✅ Testing Complete!")
print(f"{'='*70}\n")

# Model comparison summary
print(f"\n{'='*70}")
print(f"📊 Model Comparison Summary")
print(f"{'='*70}")
print(f"""
Model Characteristics:

1. ARIMA (Statistical)
   ✅ Fast, interpretable
   ✅ Good for stable trends
   ❌ Struggles with complex patterns
   
2. Prophet (Facebook)
   ✅ Handles seasonality well
   ✅ Robust to missing data
   ❌ Requires more data
   
3. XGBoost (Gradient Boosting)
   ✅ High accuracy
   ✅ Captures non-linear patterns
   ✅ Feature importance
   
4. LSTM (Deep Learning)
   ✅ Best for complex patterns
   ✅ Long-term dependencies
   ❌ Requires most data
   ❌ Slower to train
   
5. Ensemble (Combined)
   ✅ Most reliable
   ✅ Balances all models
   ✅ Highest confidence
""")