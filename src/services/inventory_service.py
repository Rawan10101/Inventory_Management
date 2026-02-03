"""
File: inventory_service.py
Description: Business logic for inventory management and demand forecasting.
Dependencies: pandas, numpy, sklearn
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class InventoryService:
    """
    Handles inventory management operations and demand forecasting.
    
    This service provides methods for analyzing inventory levels, predicting demand,
    and generating recommendations for stock optimization.
    
    Attributes:
        inventory_data (pd.DataFrame): Current inventory dataset.
        sales_data (pd.DataFrame): Historical sales dataset.
    
    Methods:
        predict_demand(item_id, period): Predicts demand for a specific item.
        calculate_reorder_point(item_id): Calculates optimal reorder point.
        identify_expiring_items(days_threshold): Identifies items near expiration.
    """
    
    def __init__(self, inventory_data: pd.DataFrame, sales_data: pd.DataFrame):
        """
        Initialize the InventoryService.
        
        Args:
            inventory_data (pd.DataFrame): Current inventory dataset.
            sales_data (pd.DataFrame): Historical sales dataset.
        """
        self.inventory_data = inventory_data
        self.sales_data = sales_data
    
    def predict_demand(self, item_id: str, period: str = 'daily') -> float:
        """
        Predicts demand for a specific item based on historical data.
        
        This is a simple moving average implementation. Students should implement
        more sophisticated forecasting methods (ARIMA, Prophet, ML models).
        
        Args:
            item_id (str): The unique identifier of the item.
            period (str): Time period for prediction ('daily', 'weekly', 'monthly').
        
        Returns:
            float: Predicted demand quantity.
        
        Raises:
            ValueError: If item_id not found in sales data.
        """
        # Filter sales data for the specific item
        item_sales = self.sales_data[self.sales_data['item_id'] == item_id]
        
        if item_sales.empty:
            raise ValueError(f"No sales data found for item: {item_id}")
        
        # Simple moving average (last 7 days for daily, etc.)
        window = 7 if period == 'daily' else 4 if period == 'weekly' else 3
        avg_demand = item_sales['quantity'].tail(window).mean()
        
        return round(avg_demand, 2)
    
    def calculate_reorder_point(self, item_id: str, lead_time_days: int = 3) -> int:
        """
        Calculates the optimal reorder point for an item.
        
        Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            item_id (str): The unique identifier of the item.
            lead_time_days (int): Number of days for supplier delivery.
        
        Returns:
            int: Recommended reorder point quantity.
        """
        daily_demand = self.predict_demand(item_id, 'daily')
        
        # Safety stock calculation (simplified - 50% of lead time demand)
        safety_stock = (daily_demand * lead_time_days) * 0.5
        
        reorder_point = (daily_demand * lead_time_days) + safety_stock
        
        return int(np.ceil(reorder_point))
    
    def identify_expiring_items(self, days_threshold: int = 7) -> pd.DataFrame:
        """
        Identifies items that are approaching expiration.
        
        Args:
            days_threshold (int): Number of days before expiration to flag items.
        
        Returns:
            pd.DataFrame: DataFrame containing items near expiration with recommendations.
        """
        # This is a placeholder - students should implement actual expiration logic
        expiring = self.inventory_data[
            self.inventory_data['days_until_expiration'] <= days_threshold
        ]
        
        return expiring.sort_values('days_until_expiration')
    
    def generate_recommendations(self, item_id: str) -> Dict[str, any]:
        """
        Generates comprehensive inventory recommendations for an item.
        
        Args:
            item_id (str): The unique identifier of the item.
        
        Returns:
            Dict[str, any]: Dictionary containing recommendations and metrics.
        """
        recommendations = {
            'item_id': item_id,
            'predicted_daily_demand': self.predict_demand(item_id, 'daily'),
            'predicted_weekly_demand': self.predict_demand(item_id, 'weekly'),
            'reorder_point': self.calculate_reorder_point(item_id),
            'status': 'optimal',  # Students should implement status logic
            'action': 'monitor'   # Students should implement action recommendations
        }
        
        return recommendations
