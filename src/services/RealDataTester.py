"""
============================================================================
REAL DATA TEST SUITE - Using Enhanced Data Loader V2.1
============================================================================
Tests the Ultimate Inventory Intelligence System V3.0 with REAL data
from your CSV files using the EnhancedDataLoader

Requirements:
- CSV files in data/Inventory_Management/
- enhanced_data_loader_v2_1.py
- ultimate_ai_solution_v3.py
============================================================================
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your modules
try:
    from enhanced_data_loader_v2_1 import EnhancedDataLoader
    from ultimate_ai_solution_v3 import UltimateInventoryIntelligence
    print("✅ Modules imported successfully\n")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure these files are in the same directory:")
    print("  - enhanced_data_loader_v2_1.py")
    print("  - ultimate_ai_solution_v3.py")
    sys.exit(1)


class RealDataTester:
    """Test suite using real CSV data through EnhancedDataLoader"""
    
    def __init__(self, data_path: str = "data/Inventory_Management"):
        """
        Initialize tester with data path
        
        Args:
            data_path: Path to directory containing CSV files
        """
        self.data_path = data_path
        self.loader = None
        self.intelligence = None
        self.sales_data = None
        self.inventory_data = None
        self.test_results = []
        
        print("="*70)
        print("🧪 REAL DATA TEST SUITE - Ultimate Inventory Intelligence V3.0")
        print("="*70)
        print(f"📁 Data Path: {data_path}\n")
    
    def log_result(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        result = {
            'test': test_name,
            'status': status,
            'details': details
        }
        self.test_results.append(result)
        print(f"{icon} {test_name}: {status}")
        if details:
            print(f"   └─ {details}\n")
    
    # ========================================================================
    # TEST 1: DATA LOADING
    # ========================================================================
    
    def test_data_loading(self):
        """Test 1: Load all data using EnhancedDataLoader"""
        print("\n" + "="*70)
        print("TEST 1: DATA LOADING WITH ENHANCED DATA LOADER")
        print("="*70 + "\n")
        
        try:
            # Initialize loader
            self.loader = EnhancedDataLoader(self.data_path)
            
            # Load daily sales (includes order processing)
            print("📊 Loading daily sales data...")
            self.sales_data = self.loader.prepare_daily_sales()
            
            if self.sales_data.empty:
                self.log_result(
                    "Data Loading - Sales",
                    "FAIL",
                    "Sales data is empty. Check if fct_orders.csv and fct_order_items.csv exist."
                )
                return False
            
            # Load inventory snapshot (uses purchase-sales method)
            print("\n📦 Loading inventory snapshot...")
            self.inventory_data = self.loader.prepare_inventory_snapshot()
            
            if self.inventory_data.empty:
                self.log_result(
                    "Data Loading - Inventory",
                    "FAIL",
                    "Inventory data is empty. Check if dim_items.csv exists."
                )
                return False
            
            # Load supporting data
            print("\n🧪 Loading Bill of Materials...")
            self.bom_data = self.loader.load_bill_of_materials()
            
            print("\n📢 Loading Campaigns...")
            self.campaign_data = self.loader.load_campaigns()
            
            print("\n🏷️  Loading Taxonomy...")
            self.taxonomy_data = self.loader.load_taxonomy_terms()
            
            # Summary
            print("\n" + "="*70)
            print("📋 DATA LOADING SUMMARY")
            print("="*70)
            print(f"Sales Records:      {len(self.sales_data):,}")
            print(f"Inventory Items:    {len(self.inventory_data):,}")
            print(f"BOM Entries:        {len(self.bom_data):,}")
            print(f"Campaigns:          {len(self.campaign_data):,}")
            print(f"Taxonomy Terms:     {len(self.taxonomy_data):,}")
            
            if 'date' in self.sales_data.columns:
                date_min = self.sales_data['date'].min()
                date_max = self.sales_data['date'].max()
                date_range = (date_max - date_min).days
                print(f"\nDate Range:         {date_min.date()} to {date_max.date()}")
                print(f"Total Days:         {date_range}")
            
            print(f"Unique Items:       {self.sales_data['item_id'].nunique():,}")
            print(f"Unique Places:      {self.sales_data['place_id'].nunique():,}")
            print("="*70 + "\n")
            
            # Validate minimum data requirements
            if len(self.sales_data) < 100:
                self.log_result(
                    "Data Loading",
                    "WARN",
                    f"Only {len(self.sales_data)} sales records. May be insufficient for ML models."
                )
            elif date_range < 90:
                self.log_result(
                    "Data Loading",
                    "WARN",
                    f"Only {date_range} days of data. Recommended: 90+ days."
                )
            else:
                self.log_result(
                    "Data Loading",
                    "PASS",
                    f"{len(self.sales_data):,} records, {date_range} days"
                )
            
            return True
            
        except FileNotFoundError as e:
            self.log_result(
                "Data Loading",
                "FAIL",
                f"File not found: {e}. Check data path: {self.data_path}"
            )
            return False
        except Exception as e:
            self.log_result(
                "Data Loading",
                "FAIL",
                f"Error: {str(e)}"
            )
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # TEST 2: DATA QUALITY VALIDATION
    # ========================================================================
    
    def test_data_quality(self):
        """Test 2: Validate data quality"""
        print("\n" + "="*70)
        print("TEST 2: DATA QUALITY VALIDATION")
        print("="*70 + "\n")
        
        if self.sales_data is None or self.inventory_data is None:
            self.log_result("Data Quality", "SKIP", "Data not loaded")
            return False
        
        try:
            # Run loader's built-in validation
            validation_results = self.loader.validate_data_quality()
            
            # Additional checks
            issues = []
            
            # Check 1: Required columns
            required_sales_cols = ['date', 'item_id', 'quantity_sold']
            missing_cols = [col for col in required_sales_cols if col not in self.sales_data.columns]
            if missing_cols:
                issues.append(f"Missing columns in sales: {missing_cols}")
            
            # Check 2: Null values
            sales_nulls = self.sales_data[required_sales_cols].isnull().sum().sum()
            if sales_nulls > 0:
                issues.append(f"{sales_nulls} null values in critical sales columns")
            
            # Check 3: Negative quantities
            if 'quantity_sold' in self.sales_data.columns:
                negative_qty = (self.sales_data['quantity_sold'] < 0).sum()
                if negative_qty > 0:
                    issues.append(f"{negative_qty} negative quantities")
            
            # Check 4: Inventory stock
            if 'current_stock' in self.inventory_data.columns:
                negative_stock = (self.inventory_data['current_stock'] < 0).sum()
                if negative_stock > 0:
                    issues.append(f"{negative_stock} items with negative stock")
            
            # Check 5: Date continuity
            if 'date' in self.sales_data.columns:
                dates = pd.to_datetime(self.sales_data['date']).sort_values()
                date_gaps = dates.diff().dt.days
                large_gaps = (date_gaps > 7).sum()
                if large_gaps > 5:
                    issues.append(f"{large_gaps} date gaps > 7 days")
            
            # Results
            if len(issues) == 0:
                self.log_result(
                    "Data Quality",
                    "PASS",
                    "All quality checks passed"
                )
                return True
            else:
                self.log_result(
                    "Data Quality",
                    "WARN",
                    f"{len(issues)} issues found"
                )
                for issue in issues:
                    print(f"   ⚠️  {issue}")
                print()
                return True  # Still continue with warnings
            
        except Exception as e:
            self.log_result("Data Quality", "FAIL", str(e))
            return False
    
    # ========================================================================
    # TEST 3: INITIALIZE AI SYSTEM
    # ========================================================================
    
    def test_ai_initialization(self):
        """Test 3: Initialize the AI intelligence system"""
        print("\n" + "="*70)
        print("TEST 3: AI SYSTEM INITIALIZATION")
        print("="*70 + "\n")
        
        if self.sales_data is None or self.inventory_data is None:
            self.log_result("AI Initialization", "SKIP", "Data not loaded")
            return False
        
        try:
            print("🧠 Initializing Ultimate Inventory Intelligence System...")
            
            self.intelligence = UltimateInventoryIntelligence(
                sales_data=self.sales_data,
                inventory_data=self.inventory_data,
                bill_of_materials=self.bom_data if not self.bom_data.empty else None,
                campaign_data=self.campaign_data if not self.campaign_data.empty else None,
                taxonomy_data=self.taxonomy_data if not self.taxonomy_data.empty else None
            )
            
            self.log_result(
                "AI Initialization",
                "PASS",
                "System initialized with all data sources"
            )
            return True
            
        except Exception as e:
            self.log_result("AI Initialization", "FAIL", str(e))
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # TEST 4: FEATURE ENGINEERING
    # ========================================================================
    
    def test_feature_engineering(self):
        """Test 4: Test feature creation with real data"""
        print("\n" + "="*70)
        print("TEST 4: FEATURE ENGINEERING (120+ FEATURES)")
        print("="*70 + "\n")
        
        if self.intelligence is None:
            self.log_result("Feature Engineering", "SKIP", "AI system not initialized")
            return False
        
        try:
            # Test on a sample
            sample_size = min(500, len(self.sales_data))
            test_sample = self.sales_data.head(sample_size).copy()
            
            print(f"📊 Testing feature creation on {sample_size} records...")
            
            original_cols = len(test_sample.columns)
            featured_df = self.intelligence.create_advanced_features(test_sample)
            new_cols = len(featured_df.columns)
            
            features_added = new_cols - original_cols
            
            print(f"\n   Original columns: {original_cols}")
            print(f"   New columns:      {new_cols}")
            print(f"   Features added:   {features_added}")
            
            # Categorize features
            feature_categories = {
                'Temporal': [col for col in featured_df.columns if any(
                    x in col for x in ['month', 'day', 'week', 'season', 'sin', 'cos', 'year', 'quarter']
                )],
                'Lag': [col for col in featured_df.columns if 'lag_' in col or 'diff_' in col or 'pct_change' in col],
                'Rolling': [col for col in featured_df.columns if 'rolling_' in col or 'ema_' in col or 'volatility' in col or 'trend_' in col],
                'Holiday': [col for col in featured_df.columns if 'holiday' in col],
                'Campaign': [col for col in featured_df.columns if 'campaign' in col],
                'Inventory': [col for col in featured_df.columns if any(
                    x in col for x in ['stock', 'shelf_life', 'expiration']
                )],
                'External': [col for col in featured_df.columns if any(
                    x in col for x in ['temperature', 'precipitation', 'weather']
                )],
                'Item': [col for col in featured_df.columns if any(
                    x in col for x in ['popularity', 'price_tier', 'velocity', 'margin', 'consistency']
                )],
                'BOM': [col for col in featured_df.columns if any(
                    x in col for x in ['ingredient', 'recipe', 'composite', 'bom']
                )]
            }
            
            print("\n   📋 Feature Categories:")
            total_categorized = 0
            for category, features in feature_categories.items():
                count = len(features)
                total_categorized += count
                if count > 0:
                    print(f"      ├─ {category:12s}: {count:3d} features")
            
            print(f"      └─ {'Other':12s}: {features_added - total_categorized:3d} features")
            
            # Validate
            if features_added < 50:
                self.log_result(
                    "Feature Engineering",
                    "WARN",
                    f"Only {features_added} features created (expected 80+)"
                )
            else:
                self.log_result(
                    "Feature Engineering",
                    "PASS",
                    f"{features_added} features created successfully"
                )
            
            return True
            
        except Exception as e:
            self.log_result("Feature Engineering", "FAIL", str(e))
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # TEST 5: MODEL TRAINING
    # ========================================================================
    
    def test_model_training(self):
        """Test 5: Train models on real data"""
        print("\n" + "="*70)
        print("TEST 5: MODEL TRAINING (9 MODELS)")
        print("="*70 + "\n")
        
        if self.intelligence is None:
            self.log_result("Model Training", "SKIP", "AI system not initialized")
            return False
        
        try:
            # Select a test item (most frequently sold)
            top_items = self.sales_data['item_id'].value_counts().head(5)
            
            if len(top_items) == 0:
                self.log_result("Model Training", "FAIL", "No items with sales data")
                return False
            
            test_item_id = top_items.index[0]
            item_sales_count = top_items.iloc[0]
            
            print(f"🎯 Testing with Item ID: {test_item_id}")
            print(f"   Sales records: {item_sales_count:,}\n")
            
            if item_sales_count < 50:
                print("⚠️  Warning: Low sales count. Some models may fail.\n")
            
            # Track model success
            model_results = {}
            
            # Traditional Models
            print("📊 TRADITIONAL MODELS (6):")
            print("-" * 70)
            
            # XGBoost
            try:
                print("   Testing XGBoost...", end=" ")
                xgb_model, xgb_metrics = self.intelligence.train_xgboost_model(test_item_id)
                if xgb_model and xgb_metrics:
                    print(f"✅ R²={xgb_metrics['r2']:.3f}, RMSE={xgb_metrics['rmse']:.2f}")
                    model_results['XGBoost'] = True
                else:
                    print("❌ Failed")
                    model_results['XGBoost'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['XGBoost'] = False
            
            # LightGBM
            try:
                print("   Testing LightGBM...", end=" ")
                lgb_model, lgb_metrics = self.intelligence.train_lightgbm_model(test_item_id)
                if lgb_model and lgb_metrics:
                    print(f"✅ R²={lgb_metrics['r2']:.3f}, RMSE={lgb_metrics['rmse']:.2f}")
                    model_results['LightGBM'] = True
                else:
                    print("❌ Failed")
                    model_results['LightGBM'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['LightGBM'] = False
            
            # GBM
            try:
                print("   Testing GBM...", end=" ")
                gbm_model, gbm_metrics = self.intelligence.train_gbm_model(test_item_id)
                if gbm_model and gbm_metrics:
                    print(f"✅ R²={gbm_metrics['r2']:.3f}, RMSE={gbm_metrics['rmse']:.2f}")
                    model_results['GBM'] = True
                else:
                    print("❌ Failed")
                    model_results['GBM'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['GBM'] = False
            
            # SARIMA
            try:
                print("   Testing SARIMA...", end=" ")
                sarima_model = self.intelligence.train_sarima_model(test_item_id)
                if sarima_model:
                    print("✅ Trained")
                    model_results['SARIMA'] = True
                else:
                    print("❌ Failed")
                    model_results['SARIMA'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['SARIMA'] = False
            
            # Prophet
            try:
                print("   Testing Prophet...", end=" ")
                prophet_model = self.intelligence.train_prophet_model(test_item_id)
                if prophet_model:
                    print("✅ Trained")
                    model_results['Prophet'] = True
                else:
                    print("❌ Failed")
                    model_results['Prophet'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['Prophet'] = False
            
            # Holt-Winters
            try:
                print("   Testing Holt-Winters...", end=" ")
                hw_model = self.intelligence.train_holtwinters_model(test_item_id)
                if hw_model:
                    print("✅ Trained")
                    model_results['HoltWinters'] = True
                else:
                    print("❌ Failed")
                    model_results['HoltWinters'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['HoltWinters'] = False
            
            # Deep Learning Models
            print("\n🧠 DEEP LEARNING MODELS (3):")
            print("-" * 70)
            
            # LSTM
            try:
                print("   Testing LSTM...", end=" ")
                lstm_model, lstm_metrics = self.intelligence.train_lstm_model(test_item_id)
                if lstm_model and lstm_metrics:
                    print(f"✅ R²={lstm_metrics['r2']:.3f}, RMSE={lstm_metrics['rmse']:.2f}")
                    model_results['LSTM'] = True
                else:
                    print("❌ Failed")
                    model_results['LSTM'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['LSTM'] = False
            
            # GRU
            try:
                print("   Testing GRU...", end=" ")
                gru_model, gru_metrics = self.intelligence.train_gru_model(test_item_id)
                if gru_model and gru_metrics:
                    print(f"✅ R²={gru_metrics['r2']:.3f}, RMSE={gru_metrics['rmse']:.2f}")
                    model_results['GRU'] = True
                else:
                    print("❌ Failed")
                    model_results['GRU'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['GRU'] = False
            
            # Transformer
            try:
                print("   Testing Transformer...", end=" ")
                trans_model, trans_metrics = self.intelligence.train_transformer_model(test_item_id)
                if trans_model and trans_metrics:
                    print(f"✅ R²={trans_metrics['r2']:.3f}, RMSE={trans_metrics['rmse']:.2f}")
                    model_results['Transformer'] = True
                else:
                    print("❌ Failed")
                    model_results['Transformer'] = False
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                model_results['Transformer'] = False
            
            # Summary
            successful_models = sum(model_results.values())
            total_models = len(model_results)
            
            print("\n" + "="*70)
            print(f"📊 MODEL TRAINING SUMMARY: {successful_models}/{total_models} models succeeded")
            print("="*70 + "\n")
            
            if successful_models >= 6:
                self.log_result(
                    "Model Training",
                    "PASS",
                    f"{successful_models}/{total_models} models trained successfully"
                )
            elif successful_models >= 3:
                self.log_result(
                    "Model Training",
                    "WARN",
                    f"Only {successful_models}/{total_models} models succeeded"
                )
            else:
                self.log_result(
                    "Model Training",
                    "FAIL",
                    f"Only {successful_models}/{total_models} models succeeded"
                )
            
            return successful_models >= 3
            
        except Exception as e:
            self.log_result("Model Training", "FAIL", str(e))
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # TEST 6: ENSEMBLE PREDICTION
    # ========================================================================
    
    def test_ensemble_prediction(self):
        """Test 6: Generate ensemble predictions"""
        print("\n" + "="*70)
        print("TEST 6: ENSEMBLE PREDICTION")
        print("="*70 + "\n")
        
        if self.intelligence is None:
            self.log_result("Ensemble Prediction", "SKIP", "AI system not initialized")
            return False
        
        try:
            # Get top items
            top_items = self.sales_data['item_id'].value_counts().head(3)
            test_item_id = top_items.index[0]
            
            print(f"🔮 Generating 7-day forecast for Item {test_item_id}...")
            print()
            
            # Generate forecast
            forecast = self.intelligence.predict_demand_ensemble(
                item_id=test_item_id,
                days_ahead=7
            )
            
            # Validate forecast structure
            assert 'ensemble_prediction' in forecast, "Missing ensemble_prediction"
            assert 'confidence_level' in forecast, "Missing confidence_level"
            assert 'predictions' in forecast, "Missing predictions"
            assert 'num_models' in forecast, "Missing num_models"
            
            # Extract results
            prediction = forecast['ensemble_prediction']
            confidence = forecast['confidence_level']
            num_models = forecast['num_models']
            dl_models = forecast.get('dl_models_used', 0)
            
            # Display results
            print("\n" + "="*70)
            print("📊 FORECAST RESULTS")
            print("="*70)
            print(f"Ensemble Prediction:  {prediction:.2f} units/day")
            print(f"Confidence Level:     {confidence:.1%}")
            print(f"Models Used:          {num_models}/9")
            print(f"DL Models Used:       {dl_models}/3")
            print("\nIndividual Predictions:")
            for model, pred in forecast['predictions'].items():
                print(f"  {model:15s}: {pred:.2f}")
            print("="*70 + "\n")
            
            # Validate results
            if prediction < 0:
                self.log_result(
                    "Ensemble Prediction",
                    "FAIL",
                    "Negative prediction"
                )
                return False
            
            if num_models == 0:
                self.log_result(
                    "Ensemble Prediction",
                    "FAIL",
                    "No models contributed to ensemble"
                )
                return False
            
            if confidence >= 0.7:
                self.log_result(
                    "Ensemble Prediction",
                    "PASS",
                    f"Prediction: {prediction:.2f}, Confidence: {confidence:.1%}, Models: {num_models}/9"
                )
            else:
                self.log_result(
                    "Ensemble Prediction",
                    "WARN",
                    f"Low confidence: {confidence:.1%}"
                )
            
            return True
            
        except Exception as e:
            self.log_result("Ensemble Prediction", "FAIL", str(e))
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # TEST 7: KITCHEN PREP CALCULATOR
    # ========================================================================
    
    def test_prep_calculator(self):
        """Test 7: Test kitchen prep calculator with real BOM"""
        print("\n" + "="*70)
        print("TEST 7: KITCHEN PREP CALCULATOR")
        print("="*70 + "\n")
        
        if self.intelligence is None:
            self.log_result("Prep Calculator", "SKIP", "AI system not initialized")
            return False
        
        if self.bom_data.empty:
            self.log_result(
                "Prep Calculator",
                "SKIP",
                "No BOM data available (dim_bill_of_materials.csv missing or empty)"
            )
            return True  # Not a failure, just skip
        
        try:
            print("👨‍🍳 Calculating ingredient requirements for 7 days...")
            
            prep_list = self.intelligence.calculate_prep_quantities(
                days_ahead=7,
                min_confidence=0.5
            )
            
            if prep_list is None or len(prep_list) == 0:
                self.log_result(
                    "Prep Calculator",
                    "WARN",
                    "No prep list generated (may need more sales history)"
                )
                return True
            
            # Display results
            print("\n" + "="*70)
            print("📋 SHOPPING LIST (Top 10)")
            print("="*70)
            print(f"{'Ingredient':<30} {'Needed':>10} {'Stock':>10} {'Order':>10} {'Unit':<8}")
            print("-"*70)
            
            for idx, row in prep_list.head(10).iterrows():
                print(f"{row['ingredient_name'][:29]:<30} "
                      f"{row['quantity_needed']:>10.1f} "
                      f"{row['current_stock']:>10.1f} "
                      f"{row['net_to_order']:>10.1f} "
                      f"{row['unit']:<8}")
            
            print("="*70)
            print(f"\nTotal Ingredients: {len(prep_list)}")
            print(f"Items to Order:    {(prep_list['net_to_order'] > 0).sum()}")
            print("="*70 + "\n")
            
            self.log_result(
                "Prep Calculator",
                "PASS",
                f"{len(prep_list)} ingredients, {(prep_list['net_to_order'] > 0).sum()} need ordering"
            )
            
            return True
            
        except Exception as e:
            self.log_result("Prep Calculator", "FAIL", str(e))
            import traceback
            print(traceback.format_exc())
            return False
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*70)
        print("🧪 STARTING COMPREHENSIVE TEST SUITE WITH REAL DATA")
        print("="*70 + "\n")
        
        tests = [
            ("Data Loading", self.test_data_loading),
            ("Data Quality", self.test_data_quality),
            ("AI Initialization", self.test_ai_initialization),
            ("Feature Engineering", self.test_feature_engineering),
            ("Model Training", self.test_model_training),
            ("Ensemble Prediction", self.test_ensemble_prediction),
            ("Prep Calculator", self.test_prep_calculator),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
                import traceback
                print(traceback.format_exc())
                results.append(False)
        
        # Print summary
        self.print_summary()
        
        return results
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print("📊 FINAL TEST SUMMARY")
        print("="*70 + "\n")
        
        if not self.test_results:
            print("No tests executed")
            return
        
        df = pd.DataFrame(self.test_results)
        
        total = len(df)
        passed = len(df[df['status'] == 'PASS'])
        failed = len(df[df['status'] == 'FAIL'])
        warnings = len(df[df['status'] == 'WARN'])
        skipped = len(df[df['status'] == 'SKIP'])
        
        print(f"Total Tests:    {total}")
        print(f"✅ Passed:      {passed}")
        print(f"❌ Failed:      {failed}")
        print(f"⚠️  Warnings:    {warnings}")
        print(f"⏭️  Skipped:     {skipped}")
        print(f"\nSuccess Rate:   {passed/total*100:.1f}%")
        
        print("\n" + "="*70)
        print("📋 DETAILED RESULTS")
        print("="*70 + "\n")
        
        for idx, row in df.iterrows():
            icon = "✅" if row['status'] == 'PASS' else "❌" if row['status'] == 'FAIL' else "⚠️" if row['status'] == 'WARN' else "⏭️"
            print(f"{icon} {row['test']}: {row['status']}")
            if row['details']:
                print(f"   └─ {row['details']}")
        
        print("\n" + "="*70)
        
        if failed == 0 and warnings == 0:
            print("🎉 PERFECT! ALL TESTS PASSED!")
            print("\nYour system is ready for production use.")
        elif failed == 0:
            print("✅ SUCCESS! All critical tests passed")
            print(f"\n{warnings} warning(s) - review but not blocking")
        elif failed <= 2:
            print("⚠️  PARTIAL SUCCESS")
            print(f"\n{failed} test(s) failed - review required")
        else:
            print("❌ MULTIPLE FAILURES")
            print(f"\n{failed} test(s) failed - needs attention")
        
        print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║        REAL DATA TEST SUITE - INVENTORY INTELLIGENCE V3.0              ║
    ║                                                                        ║
    ║  Tests with ACTUAL CSV data using EnhancedDataLoader V2.1              ║
    ║  9 Models: XGBoost, LightGBM, GBM, SARIMA, Prophet, Holt-Winters,     ║
    ║            LSTM, GRU, Transformer                                      ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # You can change the data path here if needed
    DATA_PATH = "D:/Inventory_Management/data/Inventory_Management"
    
    # Create and run tester
    tester = RealDataTester(data_path=DATA_PATH)
    results = tester.run_all_tests()
    
    # Exit code
    passed = sum([r for r in results if r])
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    exit_code = 0 if success_rate >= 80 else 1
    
    print(f"\n🏁 Test suite completed")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Exit Code: {exit_code}")
    print("="*70 + "\n")