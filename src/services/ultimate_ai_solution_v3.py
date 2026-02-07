"""
============================================================================
FRESH FLOW COMPETITION - ULTIMATE AI SOLUTION V3.5 WITH FIXED DATA PROCESSING
============================================================================
FIXED VERSION: Proper data preprocessing for ML models
CRITICAL FIXES:
1. Fixed campaign_type encoding for ML models
2. Use only recent 45-day data for training (post-holiday)
3. Proper datetime handling in campaign features
4. Added data quality validation
5. Enhanced feature selection
============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STATISTICAL & ML IMPORTS
# ============================================================================
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ============================================================================
# DEEP LEARNING IMPORTS (TENSORFLOW/KERAS)
# ============================================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️ TensorFlow not installed. Deep learning models disabled.")

class UltimateInventoryIntelligenceV3_5:
    """
    UPDATED VERSION 3.5 - FIXED DATA PROCESSING
    
    Critical Fixes:
    1. FIXED: campaign_type encoding for ML models
    2. FIXED: Use recent 45-day data only (post-holiday)
    3. FIXED: Proper datetime handling
    4. ADDED: Data quality validation
    5. ADDED: Recent trend detection
    """
    
    def __init__(self, sales_data: pd.DataFrame, inventory_data: pd.DataFrame, 
                 bill_of_materials: Optional[pd.DataFrame] = None,
                 campaign_data: Optional[pd.DataFrame] = None,
                 taxonomy_data: Optional[pd.DataFrame] = None,
                 use_recent_data: bool = True,  # NEW: Use recent data only
                 recent_days: int = 45):        # NEW: Last 45 days
        """
        Initialize with enhanced data validation
        
        Args:
            use_recent_data: If True, use only recent data for training (post-holiday)
            recent_days: Number of recent days to use (default 45 for post-holiday)
        """
        self.sales_data = sales_data.copy()
        self.inventory_data = inventory_data.copy()
        self.bom = bill_of_materials.copy() if bill_of_materials is not None else pd.DataFrame()
        self.campaigns = campaign_data.copy() if campaign_data is not None else pd.DataFrame()
        self.taxonomy = taxonomy_data.copy() if taxonomy_data is not None else pd.DataFrame()
        
        # NEW: Recent data settings
        self.use_recent_data = use_recent_data
        self.recent_days = recent_days
        
        # Campaign type encoder (FIXED: For ML models)
        self.campaign_encoder = LabelEncoder()
        
        # Model storage
        self.models = {}
        self.dl_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Deep learning parameters
        self.sequence_length = 14
        self.dl_enabled = HAS_TF
        
        # Validate and prepare data
        self._validate_data_quality()
        self._prepare_data()
        
        print("="*80)
        print("✅ ULTIMATE INVENTORY INTELLIGENCE SYSTEM V3.5 - FIXED DATA PROCESSING")
        print("="*80)
        print(f"📊 Total Sales Records: {len(self.sales_data):,}")
        print(f"📊 Recent Data Used: {'YES' if self.use_recent_data else 'NO'}")
        if self.use_recent_data:
            print(f"📊 Recent Days Window: {self.recent_days} days")
        
        # Show data quality report
        self._show_data_quality_report()
        
    def _validate_data_quality(self):
        """Validate data quality and fix common issues"""
        print("\n🔍 DATA QUALITY VALIDATION")
        print("-" * 40)
        
        # Check sales data
        if self.sales_data.empty:
            raise ValueError("Sales data is empty!")
        
        # Ensure date column exists
        if 'date' not in self.sales_data.columns:
            if 'created' in self.sales_data.columns:
                print("  ⚠️  Converting 'created' to 'date' column")
                self.sales_data['date'] = pd.to_datetime(self.sales_data['created'], unit='s').dt.date
            else:
                raise ValueError("No date column found in sales data!")
        else:
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
        
        # Check for missing dates
        date_range = (self.sales_data['date'].max() - self.sales_data['date'].min()).days + 1
        print(f"  📅 Date range: {self.sales_data['date'].min().date()} to {self.sales_data['date'].max().date()}")
        print(f"  📅 Total days: {date_range} days")
        
        # Calculate actual daily sales
        daily_sales = self.sales_data.groupby('date')['quantity_sold'].sum()
        avg_daily = daily_sales.mean()
        print(f"  📈 Average daily sales: {avg_daily:.1f} units")
        
        # Filter to recent data if requested
        if self.use_recent_data and len(self.sales_data) > 0:
            cutoff_date = self.sales_data['date'].max() - timedelta(days=self.recent_days)
            recent_mask = self.sales_data['date'] >= cutoff_date
            self.sales_data = self.sales_data[recent_mask].copy()
            
            print(f"  ✂️  Using recent {self.recent_days} days only")
            print(f"  📊 Recent records: {len(self.sales_data):,}")
            
            # Recalculate with recent data
            recent_daily = self.sales_data.groupby('date')['quantity_sold'].sum().mean()
            print(f"  📈 Recent daily average: {recent_daily:.1f} units")
            print(f"  📉 Trend vs full data: {(recent_daily/avg_daily - 1)*100:+.1f}%")
        
        # Fix campaign data for ML models
        if not self.campaigns.empty:
            print(f"\n  📢 Processing campaign data...")
            print(f"  📊 Campaign records: {len(self.campaigns):,}")
            
            # FIX: Encode campaign_type for ML models
            if 'type' in self.campaigns.columns:
                # Fill NaN with 'None'
                self.campaigns['type'] = self.campaigns['type'].fillna('None')
                # Fit encoder
                unique_types = self.campaigns['type'].unique()
                self.campaigns['type_encoded'] = pd.Categorical(self.campaigns['type']).codes
                print(f"  ✅ Encoded campaign types: {len(unique_types)} unique types")
            else:
                print(f"  ⚠️  No 'type' column in campaigns")
        
        print("-" * 40)
        print("✅ Data validation completed\n")
    
    def _show_data_quality_report(self):
        """Show detailed data quality report"""
        print("\n📋 DATA QUALITY REPORT")
        print("="*40)
        
        # Sales data summary
        print("📊 SALES DATA:")
        print(f"  • Records: {len(self.sales_data):,}")
        print(f"  • Date range: {self.sales_data['date'].min().date()} to {self.sales_data['date'].max().date()}")
        print(f"  • Unique items: {self.sales_data['item_id'].nunique()}")
        print(f"  • Total quantity sold: {self.sales_data['quantity_sold'].sum():,}")
        
        daily_avg = self.sales_data.groupby('date')['quantity_sold'].sum().mean()
        print(f"  • Average daily sales: {daily_avg:.1f}")
        
        # Inventory summary
        if not self.inventory_data.empty:
            print(f"\n📦 INVENTORY DATA:")
            print(f"  • Items: {len(self.inventory_data)}")
            print(f"  • Average stock: {self.inventory_data['current_stock'].mean():.1f}")
            print(f"  • Total value: ${self.inventory_data['total_value'].sum():,.2f}")
        
        print("="*40)
    
    # ========================================================================
    # FIXED FEATURE ENGINEERING METHODS
    # ========================================================================
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features with FIXED data processing"""
        df = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Calculate days of week BEFORE any filtering
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add month and day
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # Create temporal features
        df = self._add_temporal_features(df)
        
        # Create lag features
        df = self._add_lag_features(df)
        
        # Add campaign features WITH FIXED ENCODING
        df = self._add_campaign_features_fixed(df)
        
        # Add other features
        df = self._add_inventory_features(df)
        df = self._add_holiday_features(df)
        df = self._add_external_factors(df)
        df = self._add_item_characteristics(df)
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _add_campaign_features_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED VERSION: Campaign features with proper encoding for ML
        
        CRITICAL FIX: campaign_type_encoded must be numeric for ML models
        """
        df = df.copy()
        
        # Initialize with defaults
        df['is_campaign_active'] = 0
        df['campaign_type_encoded'] = 0  # Default to 0 (no campaign)
        df['campaign_discount_pct'] = 0
        
        if self.campaigns.empty:
            return df
        
        print("  🔧 Processing campaign features...")
        
        # Prepare campaigns with proper date handling
        campaigns = self.campaigns.copy()
        
        # Find start/end date columns
        date_cols = []
        for col in campaigns.columns:
            if 'date' in col.lower() or 'start' in col.lower() or 'end' in col.lower():
                date_cols.append(col)
        
        if len(date_cols) < 2:
            print(f"  ⚠️  Insufficient date columns in campaigns")
            return df
        
        # Use first two date columns as start/end
        start_col = date_cols[0]
        end_col = date_cols[1] if len(date_cols) > 1 else date_cols[0]
        
        # Convert to datetime
        campaigns['start_date'] = pd.to_datetime(campaigns[start_col], errors='coerce')
        campaigns['end_date'] = pd.to_datetime(campaigns[end_col], errors='coerce')
        
        # Drop rows with invalid dates
        campaigns = campaigns.dropna(subset=['start_date', 'end_date'])
        
        if len(campaigns) == 0:
            print(f"  ⚠️  No valid campaign dates found")
            return df
        
        # Create type encoding map
        type_map = {'None': 0}
        if 'type' in campaigns.columns:
            unique_types = campaigns['type'].unique()
            type_map.update({t: i+1 for i, t in enumerate(unique_types)})
        
        # Process each sales record
        for idx, row in df.iterrows():
            current_date = row['date']
            
            # Find active campaigns
            active_mask = (
                (campaigns['start_date'] <= current_date) & 
                (campaigns['end_date'] >= current_date)
            )
            
            active_campaigns = campaigns[active_mask]
            
            if len(active_campaigns) > 0:
                df.at[idx, 'is_campaign_active'] = 1
                
                # Use first campaign for type and discount
                first_campaign = active_campaigns.iloc[0]
                
                # Get campaign type (encoded)
                if 'type' in first_campaign:
                    campaign_type = str(first_campaign['type'])
                    df.at[idx, 'campaign_type_encoded'] = type_map.get(campaign_type, 0)
                
                # Get discount if available
                if 'discount_value' in first_campaign:
                    df.at[idx, 'campaign_discount_pct'] = float(first_campaign['discount_value'])
                elif 'discount' in first_campaign:
                    df.at[idx, 'campaign_discount_pct'] = float(first_campaign['discount'])
        
        print(f"  ✅ Campaign features: {df['is_campaign_active'].sum():,} active campaign days")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        df = df.copy()
        
        # Basic temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business days
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series"""
        df = df.copy()
        df = df.sort_values(['item_id', 'date'])
        
        # Group by item
        for item_id in df['item_id'].unique():
            item_mask = df['item_id'] == item_id
            
            # Create lag features
            for lag in [1, 2, 3, 7, 14]:
                df.loc[item_mask, f'lag_{lag}'] = df.loc[item_mask, 'quantity_sold'].shift(lag)
            
            # Rolling statistics
            df.loc[item_mask, 'rolling_7_mean'] = df.loc[item_mask, 'quantity_sold'].rolling(7, min_periods=1).mean()
            df.loc[item_mask, 'rolling_7_std'] = df.loc[item_mask, 'quantity_sold'].rolling(7, min_periods=1).std()
            
            # Trends
            df.loc[item_mask, 'trend_7'] = df.loc[item_mask, 'quantity_sold'].rolling(7, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        return df
    
    # ========================================================================
    # FIXED MODEL TRAINING METHODS
    # ========================================================================
    
    def train_xgboost_model_fixed(self, item_id: int):
        """
        FIXED XGBoost training with proper feature selection
        """
        try:
            # Get item data
            item_data = self.sales_data[
                self.sales_data['item_id'] == item_id
            ].sort_values('date').copy()
            
            if len(item_data) < 30:
                return None, None
            
            # Create features
            item_data = self.create_advanced_features(item_data)
            
            # Select ONLY numeric features for ML models
            numeric_cols = item_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target and ID columns
            cols_to_remove = ['quantity_sold', 'item_id', 'place_id', 'revenue', 'cost']
            feature_cols = [col for col in numeric_cols if col not in cols_to_remove]
            
            if len(feature_cols) == 0:
                return None, None
            
            X = item_data[feature_cols]
            y = item_data['quantity_sold']
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                enable_categorical=False  # IMPORTANT: No categorical features
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            return model, {
                'r2': r2,
                'rmse': rmse,
                'feature_importance': feature_importance,
                'feature_cols': feature_cols
            }
            
        except Exception as e:
            print(f"  ⚠️ XGBoost training failed for item {item_id}: {str(e)[:100]}")
            return None, None
    
    def train_lightgbm_model_fixed(self, item_id: int):
        """
        FIXED LightGBM training with proper categorical handling
        """
        try:
            item_data = self.sales_data[
                self.sales_data['item_id'] == item_id
            ].sort_values('date').copy()
            
            if len(item_data) < 30:
                return None, None
            
            item_data = self.create_advanced_features(item_data)
            
            # Select numeric features only
            numeric_cols = item_data.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_remove = ['quantity_sold', 'item_id', 'place_id', 'revenue', 'cost']
            feature_cols = [col for col in numeric_cols if col not in cols_to_remove]
            
            X = item_data[feature_cols]
            y = item_data['quantity_sold']
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # IMPORTANT: Convert to numpy arrays
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values
            
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return model, {
                'r2': r2,
                'rmse': rmse,
                'feature_cols': feature_cols
            }
            
        except Exception as e:
            print(f"  ⚠️ LightGBM training failed for item {item_id}: {str(e)[:100]}")
            return None, None
    
    # ========================================================================
    # ENHANCED PREDICTION METHOD
    # ========================================================================
    
    def predict_demand_fixed(self, item_id: int, days_ahead: int = 7, 
                           use_ensemble: bool = True) -> Dict:
        """
        Enhanced prediction with data validation
        
        Returns realistic predictions based on recent trends
        """
        print(f"\n{'='*70}")
        print(f"🔮 FORECASTING DEMAND - Item {item_id} ({days_ahead} days ahead)")
        print(f"{'='*70}")
        
        # Get historical data for this item
        item_data = self.sales_data[
            self.sales_data['item_id'] == item_id
        ].sort_values('date')
        
        if len(item_data) < 7:
            print(f"  ⚠️ Insufficient data for item {item_id}")
            return {
                'ensemble_prediction': 0,
                'confidence_level': 0,
                'predictions': {},
                'historical_average': 0,
                'recent_trend': 0
            }
        
        # Calculate historical statistics
        hist_avg = item_data['quantity_sold'].mean()
        recent_avg = item_data.tail(14)['quantity_sold'].mean()
        
        print(f"  📊 Historical average: {hist_avg:.2f}")
        print(f"  📊 Recent 14-day average: {recent_avg:.2f}")
        
        # Train models
        predictions = {}
        
        # 1. Simple baseline (recent average)
        predictions['Recent_Average'] = recent_avg
        
        # 2. XGBoost
        xgb_model, xgb_metrics = self.train_xgboot_model_fixed(item_id)
        if xgb_model and xgb_metrics:
            try:
                # Prepare features for prediction
                last_row = item_data.iloc[-1:].copy()
                last_date = last_row['date'].iloc[0]
                
                # Create future dates
                future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
                
                # Create feature dataframe for future dates
                future_rows = []
                for i, future_date in enumerate(future_dates):
                    future_row = last_row.copy()
                    future_row['date'] = future_date
                    
                    # Update temporal features
                    future_row['day_of_week'] = future_date.weekday()
                    future_row['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
                    future_row['month'] = future_date.month
                    future_row['day'] = future_date.day
                    
                    # Calculate lag features (using predicted values)
                    # This is simplified - in production you'd use a proper autoregressive approach
                    future_row['lag_1'] = recent_avg
                    future_row['lag_7'] = recent_avg
                    
                    future_rows.append(future_row)
                
                future_df = pd.concat(future_rows, ignore_index=True)
                future_df = self.create_advanced_features(future_df)
                
                # Select only the features used in training
                X_future = future_df[xgb_metrics['feature_cols']]
                
                # Predict
                xgb_preds = xgb_model.predict(X_future)
                predictions['XGBoost'] = max(0, xgb_preds.mean())
                print(f"  ✅ XGBoost: {predictions['XGBoost']:.2f}")
                
            except Exception as e:
                print(f"  ⚠️ XGBoost prediction error: {str(e)[:100]}")
        
        # 3. LightGBM
        lgb_model, lgb_metrics = self.train_lightgbm_model_fixed(item_id)
        if lgb_model and lgb_metrics:
            try:
                # Use same future_df as XGBoost
                if 'future_df' in locals():
                    future_df = self.create_advanced_features(future_df)
                    X_future = future_df[lgb_metrics['feature_cols']]
                    lgb_preds = lgb_model.predict(X_future.values)
                    predictions['LightGBM'] = max(0, lgb_preds.mean())
                    print(f"  ✅ LightGBM: {predictions['LightGBM']:.2f}")
            except Exception as e:
                print(f"  ⚠️ LightGBM prediction error: {str(e)[:100]}")
        
        # 4. SARIMA (for seasonal patterns)
        try:
            ts_data = item_data.set_index('date')['quantity_sold']
            if len(ts_data) > 30:
                model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,7))
                fitted_model = model.fit(disp=False, maxiter=50)
                sarima_pred = fitted_model.forecast(steps=days_ahead).mean()
                predictions['SARIMA'] = max(0, sarima_pred)
                print(f"  ✅ SARIMA: {sarima_pred:.2f}")
        except Exception as e:
            print(f"  ⚠️ SARIMA failed: {str(e)[:100]}")
        
        # Ensemble prediction
        if predictions:
            # Weight recent average higher (40%) since it's most relevant
            weights = {
                'Recent_Average': 0.4,
                'XGBoost': 0.25,
                'LightGBM': 0.25,
                'SARIMA': 0.1
            }
            
            # Calculate weighted ensemble
            ensemble_pred = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0.1)
                ensemble_pred += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            # Calculate confidence
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                std_dev = np.std(pred_values)
                mean_pred = np.mean(pred_values)
                confidence = max(0, 1 - (std_dev / (mean_pred + 1)))
            else:
                confidence = 0.7  # Default confidence
            
            print(f"\n{'='*70}")
            print(f"🎯 ENSEMBLE PREDICTION: {ensemble_pred:.2f} units/day")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"📈 Based on: {len(predictions)} models")
            print(f"📊 Recent trend: {recent_avg:.1f} (vs historical {hist_avg:.1f})")
            print(f"{'='*70}")
            
            return {
                'ensemble_prediction': ensemble_pred,
                'confidence_level': confidence,
                'predictions': predictions,
                'historical_average': hist_avg,
                'recent_average': recent_avg,
                'trend_change_pct': (recent_avg/hist_avg - 1) * 100
            }
        else:
            print(f"  ⚠️ All models failed, using recent average")
            return {
                'ensemble_prediction': recent_avg,
                'confidence_level': 0.5,
                'predictions': {'Recent_Average': recent_avg},
                'historical_average': hist_avg,
                'recent_average': recent_avg
            }


# ============================================================================
# UPDATED PIPELINE INTEGRATION
# ============================================================================

def run_updated_pipeline(data_dir: str, output_dir: str, use_recent_data: bool = True):
    """
    Updated pipeline with fixed data processing
    """
    print("="*80)
    print("🚀 RUNNING UPDATED PIPELINE V3.5 - WITH FIXED DATA PROCESSING")
    print("="*80)
    
    # Load data (use your existing data loading methods)
    from src.services.data_loader import EnhancedDataLoader
    
    data_loader = EnhancedDataLoader(data_dir)
    sales_data = data_loader.get_daily_sales()
    inventory_data = data_loader.get_inventory_snapshot()
    
    # Check if we have campaigns data
    campaign_data = pd.DataFrame()
    try:
        campaign_data = data_loader.load_campaigns()
        print(f"✅ Loaded campaign data: {len(campaign_data):,} records")
    except:
        print("ℹ️ No campaign data available")
    
    # Initialize updated intelligence system
    intelligence = UltimateInventoryIntelligenceV3_5(
        sales_data=sales_data,
        inventory_data=inventory_data,
        campaign_data=campaign_data,
        use_recent_data=use_recent_data,
        recent_days=45  # Use last 45 days for post-holiday predictions
    )
    
    # Get unique items
    unique_items = sales_data['item_id'].unique()
    print(f"\n📦 Forecasting for {len(unique_items)} unique items")
    
    # Generate predictions
    forecasts = []
    for item_id in unique_items[:50]:  # Limit to 50 items for demo
        forecast = intelligence.predict_demand_fixed(item_id, days_ahead=7)
        
        # Get item name
        item_name = inventory_data[inventory_data['item_id'] == item_id]['title'].iloc[0] \
            if not inventory_data.empty else f"Item_{item_id}"
        
        forecasts.append({
            'item_id': item_id,
            'item_name': item_name,
            'predicted_daily_demand': forecast['ensemble_prediction'],
            'confidence': forecast['confidence_level'],
            'historical_average': forecast['historical_average'],
            'recent_average': forecast['recent_average'],
            'trend_change_pct': forecast.get('trend_change_pct', 0),
            'num_models_used': len(forecast['predictions'])
        })
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame(forecasts)
    
    # Calculate summary statistics
    total_predicted = forecast_df['predicted_daily_demand'].sum()
    avg_confidence = forecast_df['confidence'].mean()
    avg_trend_change = forecast_df['trend_change_pct'].mean()
    
    print(f"\n📊 FORECAST SUMMARY:")
    print(f"   • Total predicted daily demand: {total_predicted:.1f} units")
    print(f"   • Average confidence: {avg_confidence:.1%}")
    print(f"   • Average trend change: {avg_trend_change:+.1f}%")
    
    # Save forecasts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    forecast_path = Path(output_dir) / f"forecast_{timestamp}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    
    print(f"\n💾 Forecast saved to: {forecast_path}")
    print("="*80)
    
    return {
        'forecasts': forecast_df,
        'summary': {
            'total_predicted_demand': total_predicted,
            'avg_confidence': avg_confidence,
            'forecast_date': datetime.now().strftime("%Y-%m-%d"),
            'use_recent_data': use_recent_data
        }
    }


if __name__ == "__main__":
    # Example usage
    data_dir = "data/Inventory_Management"
    output_dir = "reports"
    
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║          FRESH FLOW - UPDATED MODEL V3.5 WITH FIXED DATA               ║
    ║                                                                        ║
    ║  CRITICAL FIXES APPLIED:                                               ║
    ║  ✅ Fixed campaign_type encoding for ML models                         ║
    ║  ✅ Use recent 45-day data (post-holiday trends)                       ║
    ║  ✅ Proper feature selection for ML                                    ║
    ║  ✅ Enhanced data validation                                           ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run with recent data only (post-holiday)
    results = run_updated_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        use_recent_data=True  # CRITICAL: Use recent data only!
    )