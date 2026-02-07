"""
File: prep_calculator.py
Description: Calculate optimal prep quantities based on demand forecasts and BOM
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class PrepCalculator:
    """
    Calculates kitchen prep quantities using BOM and demand forecasts
    """
    
    def __init__(self, bom_df: pd.DataFrame, inventory_df: pd.DataFrame):
        self.bom = self._normalize_bom(bom_df)
        self.inventory = inventory_df

    def _normalize_bom(self, bom_df: pd.DataFrame) -> pd.DataFrame:
        if bom_df is None or bom_df.empty:
            return pd.DataFrame()

        df = bom_df.copy()

        # Common alternative column mappings
        if "menu_item_id" not in df.columns and "parent_sku_id" in df.columns:
            df["menu_item_id"] = df["parent_sku_id"]
        if "ingredient_id" not in df.columns and "sku_id" in df.columns:
            df["ingredient_id"] = df["sku_id"]
        if "quantity_per_serving" not in df.columns and "quantity" in df.columns:
            df["quantity_per_serving"] = df["quantity"]

        # Fallback defaults
        if "ingredient_name" not in df.columns:
            df["ingredient_name"] = df.get("ingredient_id", pd.Series(["Unknown"] * len(df))).astype(str)
        if "stock_unit" not in df.columns:
            df["stock_unit"] = "unit"
        if "unit_cost" not in df.columns:
            df["unit_cost"] = 0.0
        if "shelf_life_days" not in df.columns:
            df["shelf_life_days"] = 3

        required = {"menu_item_id", "ingredient_id", "quantity_per_serving"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        return df
    
    def calculate_prep_quantities(self, 
                                  demand_forecast: pd.DataFrame, 
                                  prep_date: datetime,
                                  prep_horizon_days: int = 1) -> pd.DataFrame:
        """
        Calculate ingredient prep needs based on forecasted menu item demand
        """
        if demand_forecast is None or demand_forecast.empty:
            return pd.DataFrame()
        if self.bom is None or self.bom.empty:
            return pd.DataFrame()
        if "item_id" not in demand_forecast.columns or "predicted_daily_demand" not in demand_forecast.columns:
            return pd.DataFrame()

        prep_list = []
        
        # Group forecast by menu item
        for item_id, forecast_row in demand_forecast.iterrows():
            menu_item_id = forecast_row['item_id']
            predicted_orders = forecast_row['predicted_daily_demand'] * prep_horizon_days
            
            # Get ingredients from BOM
            ingredients = self.bom[self.bom['menu_item_id'] == menu_item_id]
            
            for _, ingredient in ingredients.iterrows():
                ingredient_id = ingredient['ingredient_id']
                qty_per_serving = ingredient['quantity_per_serving']
                shelf_life = ingredient['shelf_life_days']
                
                # Total ingredient needed
                total_needed = qty_per_serving * predicted_orders
                
                # Current stock
                current_stock_row = self.inventory[
                    self.inventory['item_id'] == ingredient_id
                ]
                
                if len(current_stock_row) > 0:
                    current_stock = current_stock_row.iloc[0]['quantity_on_hand']
                else:
                    current_stock = 0
                
                # Calculate prep quantity
                # Don't prep more than shelf life allows
                max_prepable = self._calculate_max_prepable(
                    predicted_orders / prep_horizon_days,  # Daily rate
                    shelf_life,
                    qty_per_serving
                )
                
                prep_needed = max(total_needed - current_stock, 0)
                prep_quantity = min(prep_needed, max_prepable)
                
                # Determine priority
                if shelf_life <= 1:
                    priority = 'critical'
                elif shelf_life <= 2:
                    priority = 'high'
                else:
                    priority = 'normal'
                
                prep_list.append({
                    'prep_date': prep_date,
                    'menu_item_id': menu_item_id,
                    'menu_item_name': forecast_row.get('item_name', 'Unknown'),
                    'ingredient_id': ingredient_id,
                    'ingredient_name': ingredient['ingredient_name'],
                    'predicted_orders': predicted_orders,
                    'quantity_per_serving': qty_per_serving,
                    'total_needed': total_needed,
                    'current_stock': current_stock,
                    'prep_quantity': prep_quantity,
                    'unit': ingredient['stock_unit'],
                    'shelf_life_days': shelf_life,
                    'priority': priority,
                    'estimated_cost': prep_quantity * ingredient['unit_cost']
                })
        
        prep_df = pd.DataFrame(prep_list)

        if prep_df.empty:
            return prep_df
        
        # Aggregate by ingredient (multiple menu items may use same ingredient)
        aggregated_prep = prep_df.groupby(['ingredient_id', 'ingredient_name', 'unit', 'shelf_life_days', 'priority']).agg({
            'prep_quantity': 'sum',
            'current_stock': 'first',
            'estimated_cost': 'sum'
        }).reset_index()
        
        # Sort by priority
        priority_order = {'critical': 1, 'high': 2, 'normal': 3}
        aggregated_prep['priority_rank'] = aggregated_prep['priority'].map(priority_order)
        aggregated_prep = aggregated_prep.sort_values('priority_rank')
        
        return aggregated_prep
    
    def _calculate_max_prepable(self, daily_demand: float, shelf_life_days: int, 
                                qty_per_serving: float) -> float:
        """
        Calculate maximum quantity that can be prepped given shelf life constraints
        """
        # Can only prep what will be used within shelf life
        max_days_to_prep = min(shelf_life_days, 3)  # Never prep more than 3 days ahead
        max_quantity = daily_demand * max_days_to_prep * qty_per_serving
        
        return max_quantity
