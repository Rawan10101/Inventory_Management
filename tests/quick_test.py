"""
File: quick_test.py
Description: Quick test script for expiration_manager.py
Run from project root: py tests\quick_test.py
"""

import sys
import os

# Add the src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import pandas as pd
from models.expiration_manager import ExpirationManager, PromotionOptimizer

print("=" * 60)
print("QUICK TEST - Expiration Manager")
print("=" * 60)

# Mock forecaster
class MockForecaster:
    pass

# Sample inventory data
print("\n📦 Creating sample inventory data...")
inventory = pd.DataFrame({
    'item_id': [1, 2, 3],
    'item_name': ['Organic Milk', 'Greek Yogurt', 'Whole Wheat Bread'],
    'quantity_on_hand': [50, 30, 40],
    'days_until_expiration': [1, 3, 7],
    'unit_cost': [4.0, 3.5, 2.5],
    'total_value': [200.0, 105.0, 100.0]
})

print(inventory.to_string(index=False))

# Sample demand predictions
demand = pd.DataFrame({
    'item_id': [1, 2, 3],
    'predicted_daily_demand': [10.0, 8.0, 12.0]
})

# Sample order history
orders = pd.DataFrame({
    'order_id': [1, 1, 2, 2, 3, 3],
    'item_id': [1, 3, 1, 2, 2, 3],
    'quantity': [1, 1, 1, 1, 1, 1]
})

# Test ExpirationManager
print("\n🧪 Testing ExpirationManager...")
manager = ExpirationManager(MockForecaster())
print("✓ Manager initialized")

# Prioritize inventory
prioritized = manager.prioritize_inventory(inventory, demand)
print("✓ Inventory prioritized")

# Get recommendations
recommendations = manager.recommend_actions(prioritized)
print("✓ Recommendations generated")

print("\n💡 RECOMMENDATIONS:")
print("-" * 60)
display_cols = ['item_name', 'days_until_expiration', 'risk_category', 
                'recommended_action', 'discount_percentage']
print(recommendations[display_cols].to_string(index=False))

# Test PromotionOptimizer
print("\n🎁 Testing PromotionOptimizer...")
optimizer = PromotionOptimizer(orders)
print("✓ Optimizer initialized")

at_risk_items = recommendations[
    recommendations['risk_category'].isin(['critical', 'high'])
]

if len(at_risk_items) > 0:
    bundles = optimizer.create_bundle_promotions(at_risk_items)
    print(f"✓ Created {len(bundles)} bundle promotions")
    
    if bundles:
        print("\n🎁 BUNDLE PROMOTIONS:")
        print("-" * 60)
        for bundle in bundles:
            print(f"  • {bundle['bundle_name']}")
            print(f"    Discount: {bundle['discount_percentage']*100:.0f}%")
            print(f"    Valid until: {bundle['valid_until'].strftime('%Y-%m-%d')}")
else:
    print("  No at-risk items requiring bundles")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
