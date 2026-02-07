"""
File: expiration_manager.py
Description: FEFO-based inventory management and promotion recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class ExpirationManager:
    """
    Manages inventory prioritization based on expiration dates [web:7][web:10]
    """
    
    def __init__(self, demand_forecaster):
        self.demand_forecaster = demand_forecaster
        self.risk_thresholds = {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }
    
    def calculate_risk_score(self, row: pd.Series, predicted_demand: float) -> float:
        """
        Calculate expiration risk score for an item
        """
        days_to_expiry = row['days_until_expiration']
        current_stock = row['quantity_on_hand']
        unit_cost = row['unit_cost']
        
        # Avoid division by zero
        if predicted_demand <= 0:
            predicted_demand = 0.1
        
        # Days of inventory (how many days will this stock last)
        days_of_inventory = current_stock / predicted_demand
        
        # Risk components
        # 1. Expiry urgency (0-1): higher as expiry approaches
        if days_to_expiry <= 0:
            expiry_urgency = 1.0
        elif days_to_expiry >= 7:
            expiry_urgency = 0.0
        else:
            expiry_urgency = (7 - days_to_expiry) / 7
        
        # 2. Overstocking risk (0-1): stock exceeds sellable time
        if days_of_inventory > days_to_expiry and days_to_expiry > 0:
            overstocking_risk = min((days_of_inventory - days_to_expiry) / days_of_inventory, 1.0)
        else:
            overstocking_risk = 0.0
        
        # 3. Value at risk (normalized)
        value_at_risk = unit_cost * current_stock
        value_at_risk_normalized = min(value_at_risk / 1000, 1.0)  # Normalize to 0-1
        
        # Weighted composite score
        risk_score = (
            0.40 * expiry_urgency +
            0.35 * overstocking_risk +
            0.25 * value_at_risk_normalized
        )
        
        return risk_score
    
    def prioritize_inventory(self, inventory_df: pd.DataFrame, 
                            demand_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Prioritize inventory items based on expiration risk
        """
        # Merge inventory with predictions
        merged = inventory_df.merge(
            demand_predictions[['item_id', 'predicted_daily_demand']],
            on='item_id',
            how='left'
        )
        
        # Fill missing predictions with average
        merged['predicted_daily_demand'].fillna(merged['predicted_daily_demand'].mean(), inplace=True)
        
        # Calculate risk scores
        merged['risk_score'] = merged.apply(
            lambda row: self.calculate_risk_score(row, row['predicted_daily_demand']),
            axis=1
        )
        
        # Assign risk categories
        def assign_risk_category(score):
            if score >= self.risk_thresholds['critical']:
                return 'critical'
            elif score >= self.risk_thresholds['high']:
                return 'high'
            elif score >= self.risk_thresholds['medium']:
                return 'medium'
            else:
                return 'low'
        
        merged['risk_category'] = merged['risk_score'].apply(assign_risk_category)
        
        # Calculate days of inventory
        merged['days_of_inventory'] = merged['quantity_on_hand'] / merged['predicted_daily_demand'].replace(0, 0.1)
        
        # Sort by risk score
        prioritized = merged.sort_values('risk_score', ascending=False)
        
        return prioritized
    
    def recommend_actions(self, prioritized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate action recommendations for at-risk items
        """
        recommendations = []
        
        for _, row in prioritized_df.iterrows():
            risk_score = row['risk_score']
            risk_category = row['risk_category']
            days_to_expiry = row['days_until_expiration']
            
            # Action logic
            if risk_category == 'critical':
                action = 'urgent_discount'
                discount_pct = 0.40  # 40% off
                priority = 1
            elif risk_category == 'high':
                action = 'bundle_promotion'
                discount_pct = 0.25  # 25% off
                priority = 2
            elif risk_category == 'medium':
                action = 'featured_item'
                discount_pct = 0.15  # 15% off
                priority = 3
            else:
                action = 'monitor'
                discount_pct = 0.0
                priority = 4
            
            recommendations.append({
                'item_id': row['item_id'],
                'item_name': row['item_name'],
                'current_stock': row['quantity_on_hand'],
                'days_until_expiration': days_to_expiry,
                'predicted_daily_demand': row['predicted_daily_demand'],
                'days_of_inventory': row['days_of_inventory'],
                'risk_score': risk_score,
                'risk_category': risk_category,
                'recommended_action': action,
                'discount_percentage': discount_pct,
                'priority': priority,
                'estimated_revenue_loss': row['total_value'] if days_to_expiry <= 0 else 0,
                'potential_recovery': row['total_value'] * (1 - discount_pct) if action != 'monitor' else 0
            })
        
        return pd.DataFrame(recommendations)


class PromotionOptimizer:
    """
    Creates intelligent promotions for near-expiry items
    """
    
    def __init__(self, order_history_df: pd.DataFrame):
        self.order_history = order_history_df
        self._build_complementary_matrix()
    
    def _build_complementary_matrix(self):
        """
        Build item-item complementary purchase matrix
        """
        # Find items frequently bought together
        order_items = self.order_history.groupby(['order_id', 'item_id']).size().reset_index(name='count')
        
        # Self-join to find co-purchases
        self.complementary_pairs = order_items.merge(
            order_items, 
            on='order_id', 
            suffixes=('_1', '_2')
        )
        self.complementary_pairs = self.complementary_pairs[
            self.complementary_pairs['item_id_1'] != self.complementary_pairs['item_id_2']
        ]
        
        # Calculate co-purchase frequency
        self.complementary_scores = self.complementary_pairs.groupby(
            ['item_id_1', 'item_id_2']
        ).size().reset_index(name='copurchase_count')
    
    def find_complementary_items(self, item_id: int, top_n: int = 3) -> List[int]:
        """
        Find items commonly purchased with this item
        """
        complements = self.complementary_scores[
            self.complementary_scores['item_id_1'] == item_id
        ].sort_values('copurchase_count', ascending=False).head(top_n)
        
        return complements['item_id_2'].tolist()
    
    def create_bundle_promotions(self, at_risk_items: pd.DataFrame) -> List[Dict]:
        """
        Create bundle promotions for at-risk items
        """
        bundles = []
        
        for _, item in at_risk_items.iterrows():
            if item['recommended_action'] in ['urgent_discount', 'bundle_promotion']:
                complements = self.find_complementary_items(item['item_id'], top_n=2)
                
                if len(complements) >= 1:
                    bundle = {
                        'bundle_id': f"BUNDLE_{item['item_id']}_{datetime.now().strftime('%Y%m%d')}",
                        'bundle_name': f"Fresh Deal: {item['item_name']} Bundle",
                        'primary_item_id': item['item_id'],
                        'primary_item_name': item['item_name'],
                        'complementary_items': complements,
                        'discount_percentage': item['discount_percentage'],
                        'valid_from': datetime.now(),
                        'valid_until': datetime.now() + timedelta(days=min(item['days_until_expiration'], 3)),
                        'priority': item['priority'],
                        'estimated_units_to_move': item['current_stock'] * 0.6,  # Expect to move 60%
                        'potential_revenue': item['potential_recovery']
                    }
                    bundles.append(bundle)
        
        return bundles