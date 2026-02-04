from src.models.data_loader import DataLoader
from src.services.inventory_service import InventoryService
import pandas as pd

# Load data
loader = DataLoader(r"D:\Inventory_Management\data\Inventory_Management")
daily_sales = loader.prepare_daily_sales()
inventory_snapshot = loader.prepare_inventory_snapshot()

# Initialize service
service = InventoryService(inventory_data=inventory_snapshot, sales_data=daily_sales)

# Pick a sample item (preferably one with inventory managed)
sample_items = daily_sales[daily_sales['manage_inventory'] == 1]['item_id'].unique()

if len(sample_items) > 0:
    sample_item = sample_items[0]
    
    print(f"\nTesting item: {sample_item}")
    print("=" * 60)
    
    try:
        demand = service.predict_demand(sample_item)
        print(f"Predicted daily demand: {demand}")
        
        reorder = service.calculate_reorder_point(sample_item)
        print(f"Reorder point: {reorder}")
        
        recommendations = service.generate_recommendations(sample_item)
        print(f"\nRecommendations:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
else:
    print("No inventory-managed items found in sales data")