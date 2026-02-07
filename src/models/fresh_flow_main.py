"""
File: fresh_flow_main.py
Description: Main orchestration module that integrates all components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import all modules - UPDATED to use DataLoader
from data_loader import DataLoader
from demand_forecasting import DemandForecaster
from expiration_manager import ExpirationManager, PromotionOptimizer
from prep_calculator import PrepCalculator

class FreshFlowSystem:
    """
    Main Fresh Flow system integrating all components
    """
    
    def __init__(self, csv_directory: str):
        """
        Initialize the Fresh Flow system
        
        Args:
            csv_directory: Path to folder containing CSV files
        """
        self.csv_directory = csv_directory
        self.data_loader = DataLoader(csv_directory)  # ✅ Changed from DataPreparation to DataLoader
        
        # Models will be initialized after training
        self.daily_forecaster = None
        self.weekly_forecaster = None
        self.monthly_forecaster = None
        self.expiration_manager = None
        self.prep_calculator = None
        self.promotion_optimizer = None
        
        # Data storage
        self.demand_df = None
        self.inventory_df = None
        self.bom_df = None
    
    def initialize_and_train(self):
        """
        Complete system initialization and model training pipeline
        """
        print("="*60)
        print("FRESH FLOW SYSTEM - INITIALIZATION")
        print("="*60)
        
        # Step 1: Load CSV data using DataLoader
        print("\nStep 1: Loading CSV data...")
        
        # Use DataLoader methods to prepare data
        self.demand_df = self.data_loader.prepare_daily_sales()
        self.inventory_df = self.data_loader.prepare_inventory_snapshot()
        
        # BOM data (if you have a bill of materials CSV, load it here)
        # For now, create empty DataFrame if you don't have BOM data
        try:
            self.bom_df = self.data_loader.load_csv("bom.csv")  # If you have BOM file
        except FileNotFoundError:
            print("No BOM file found, creating empty DataFrame")
            self.bom_df = pd.DataFrame()  # Empty DataFrame as placeholder
        
        # Rename columns to match expected format for downstream models
        if 'date' in self.demand_df.columns:
            self.demand_df = self.demand_df.rename(columns={'date': 'order_date'})
        if 'title' in self.demand_df.columns:
            self.demand_df = self.demand_df.rename(columns={'title': 'item_name'})
        if 'title_place' in self.demand_df.columns:
            self.demand_df = self.demand_df.rename(columns={'title_place': 'merchant_name'})
        
        # For inventory, ensure proper column names
        if 'title' in self.inventory_df.columns:
            self.inventory_df = self.inventory_df.rename(columns={'title': 'item_name'})
        if 'current_stock' in self.inventory_df.columns:
            self.inventory_df = self.inventory_df.rename(columns={'current_stock': 'quantity_on_hand'})
        
        # Add missing columns if needed for expiration manager
        if 'days_until_expiration' not in self.inventory_df.columns:
            # Default to 7 days if not specified
            self.inventory_df['days_until_expiration'] = 7
        if 'unit_cost' not in self.inventory_df.columns:
            # Estimate from revenue/quantity if available, else default
            self.inventory_df['unit_cost'] = 10.0  # Default placeholder
        if 'total_value' not in self.inventory_df.columns:
            self.inventory_df['total_value'] = self.inventory_df['quantity_on_hand'] * self.inventory_df['unit_cost']
        
        print(f"✓ Loaded {len(self.demand_df)} demand records")
        print(f"✓ Loaded {len(self.inventory_df)} inventory records")
        
        # Step 2: Train forecasting models
        print("\n" + "="*60)
        print("Step 2: Training demand forecasting models...")
        print("="*60)
        
        # Daily forecaster
        self.daily_forecaster = DemandForecaster(forecast_horizon='daily')
        self.daily_forecaster.train(self.demand_df, target_col='quantity_sold')
        
        # Save model
        import os
        os.makedirs('models', exist_ok=True)
        self.daily_forecaster.save_model('models/daily_forecaster.pkl')
        
        # Evaluate
        print("\nEvaluating daily forecaster on recent data...")
        recent_data = self.demand_df[self.demand_df['order_date'] >= 
                                    self.demand_df['order_date'].max() - timedelta(days=30)]
        self.daily_forecaster.evaluate(recent_data)
        
        # Initialize other components
        print("\n" + "="*60)
        print("Step 3: Initializing inventory management components...")
        print("="*60)
        
        self.expiration_manager = ExpirationManager(self.daily_forecaster)
        
        # Only initialize PrepCalculator if we have BOM data
        if not self.bom_df.empty:
            self.prep_calculator = PrepCalculator(self.bom_df, self.inventory_df)
        else:
            print("⚠ No BOM data available - PrepCalculator disabled")
            self.prep_calculator = None
        
        self.promotion_optimizer = PromotionOptimizer(self.demand_df)
        
        print("\n✓ Fresh Flow System successfully initialized and trained!")
        
    def generate_daily_forecast(self, target_date: datetime = None) -> pd.DataFrame:
        """
        Generate demand forecast for a specific date
        """
        if target_date is None:
            target_date = datetime.now() + timedelta(days=1)
        
        print(f"\nGenerating forecast for {target_date.strftime('%Y-%m-%d')}...")
        
        # Prepare feature data for target date
        latest_data = self.demand_df[self.demand_df['order_date'] == 
                                      self.demand_df['order_date'].max()].copy()
        
        # Make predictions
        predictions = self.daily_forecaster.predict(latest_data)
        
        # Create forecast dataframe
        forecast_df = latest_data[['item_id', 'item_name', 'place_id', 'merchant_name']].copy()
        forecast_df['forecast_date'] = target_date
        forecast_df['predicted_daily_demand'] = predictions
        forecast_df['predicted_daily_demand'] = forecast_df['predicted_daily_demand'].round(2)
        
        return forecast_df
    
    def analyze_expiration_risks(self) -> tuple:
        """
        Analyze inventory for expiration risks and generate recommendations
        """
        print("\nAnalyzing expiration risks...")
        
        # Get latest inventory
        latest_inventory = self.inventory_df[
            self.inventory_df['report_date'] == self.inventory_df['report_date'].max()
        ].copy()
        
        # Generate demand predictions for current inventory items
        demand_forecast = self.generate_daily_forecast()
        
        # Prioritize inventory
        prioritized = self.expiration_manager.prioritize_inventory(
            latest_inventory, 
            demand_forecast
        )
        
        # Generate action recommendations
        recommendations = self.expiration_manager.recommend_actions(prioritized)
        
        # Filter to at-risk items only
        at_risk = recommendations[recommendations['risk_category'].isin(['critical', 'high', 'medium'])]
        
        print(f"\n✓ Found {len(at_risk)} items at risk of expiration")
        print(f"  - Critical: {len(at_risk[at_risk['risk_category']=='critical'])}")
        print(f"  - High: {len(at_risk[at_risk['risk_category']=='high'])}")
        print(f"  - Medium: {len(at_risk[at_risk['risk_category']=='medium'])}")
        
        return prioritized, recommendations, at_risk
    
    def generate_promotions(self, at_risk_items: pd.DataFrame) -> List[Dict]:
        """
        Generate promotion bundles for at-risk items
        """
        print("\nGenerating promotion bundles...")
        
        bundles = self.promotion_optimizer.create_bundle_promotions(at_risk_items)
        
        print(f"✓ Created {len(bundles)} promotional bundles")
        
        return bundles
    
    def calculate_prep_needs(self, prep_date: datetime = None, 
                            prep_horizon_days: int = 1) -> pd.DataFrame:
        """
        Calculate kitchen prep quantities needed
        """
        if self.prep_calculator is None:
            print("⚠ PrepCalculator not available (no BOM data)")
            return pd.DataFrame()
        
        if prep_date is None:
            prep_date = datetime.now() + timedelta(days=1)
        
        print(f"\nCalculating prep needs for {prep_date.strftime('%Y-%m-%d')}...")
        
        # Get demand forecast
        demand_forecast = self.generate_daily_forecast(prep_date)
        
        # Calculate prep quantities
        prep_plan = self.prep_calculator.calculate_prep_quantities(
            demand_forecast,
            prep_date,
            prep_horizon_days
        )
        
        print(f"✓ Prep plan generated for {len(prep_plan)} ingredients")
        if len(prep_plan) > 0:
            print(f"  Total estimated cost: DKK {prep_plan['estimated_cost'].sum():.2f}")
        
        return prep_plan
    
    def generate_daily_report(self) -> Dict:
        """
        Generate comprehensive daily operations report
        """
        print("\n" + "="*60)
        print("GENERATING DAILY OPERATIONS REPORT")
        print("="*60)
        
        report = {}
        
        # 1. Demand forecast
        forecast = self.generate_daily_forecast()
        report['demand_forecast'] = forecast
        
        # 2. Expiration analysis
        prioritized, recommendations, at_risk = self.analyze_expiration_risks()
        report['inventory_prioritization'] = prioritized
        report['action_recommendations'] = recommendations
        report['at_risk_items'] = at_risk
        
        # 3. Promotions
        if len(at_risk) > 0:
            promotions = self.generate_promotions(at_risk)
            report['promotions'] = promotions
        else:
            report['promotions'] = []
        
        # 4. Prep plan
        prep_plan = self.calculate_prep_needs()
        report['prep_plan'] = prep_plan
        
        # 5. Summary metrics
        report['summary'] = {
            'forecast_date': datetime.now() + timedelta(days=1),
            'total_predicted_demand': forecast['predicted_daily_demand'].sum(),
            'at_risk_items_count': len(at_risk),
            'critical_items_count': len(at_risk[at_risk['risk_category']=='critical']),
            'potential_waste_value': at_risk['estimated_revenue_loss'].sum() if len(at_risk) > 0 else 0,
            'potential_recovery_value': at_risk['potential_recovery'].sum() if len(at_risk) > 0 else 0,
            'promotions_created': len(report['promotions']),
            'prep_ingredients_count': len(prep_plan),
            'prep_total_cost': prep_plan['estimated_cost'].sum() if len(prep_plan) > 0 else 0
        }
        
        print("\n" + "="*60)
        print("REPORT SUMMARY")
        print("="*60)
        for key, value in report['summary'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        return report
    
    def export_report(self, report: Dict, output_dir: str = 'reports'):
        """
        Export report to CSV files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export each component
        report['demand_forecast'].to_csv(f'{output_dir}/forecast_{timestamp}.csv', index=False)
        report['action_recommendations'].to_csv(f'{output_dir}/recommendations_{timestamp}.csv', index=False)
        
        if len(report['prep_plan']) > 0:
            report['prep_plan'].to_csv(f'{output_dir}/prep_plan_{timestamp}.csv', index=False)
        
        # Export promotions
        if len(report['promotions']) > 0:
            pd.DataFrame(report['promotions']).to_csv(f'{output_dir}/promotions_{timestamp}.csv', index=False)
        
        # Export summary
        pd.DataFrame([report['summary']]).to_csv(f'{output_dir}/summary_{timestamp}.csv', index=False)
        
        print(f"\n✓ Reports exported to {output_dir}/ directory")

# Example usage
if __name__ == "__main__":
    # UPDATE THIS PATH to your CSV data directory
    CSV_PATH = r"D:\Extracurricular\Competetions\Deliotte\Inventory_Management\data"
    
    # Initialize system
    system = FreshFlowSystem(csv_directory=CSV_PATH)
    
    # Train all models
    system.initialize_and_train()
    
    # Generate daily report
    report = system.generate_daily_report()
    
    # Export results
    system.export_report(report)
    
    print("\n✓ Fresh Flow System operational!")