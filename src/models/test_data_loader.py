import os
from data_loader import DataLoader

# Absolute project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(BASE_DIR, "data", "Inventory_Management")

print("Resolved data path:", DATA_PATH)

loader = DataLoader(data_path=DATA_PATH)

orders = loader.load_orders()
order_items = loader.load_order_items()
items = loader.load_items()
places = loader.load_places()

print("Orders shape:", orders.shape)
print("Order items shape:", order_items.shape)
print("Items shape:", items.shape)
print("Places shape:", places.shape)

daily_sales = loader.prepare_daily_sales()

print("\nDaily sales preview:")
print(daily_sales.head())

print("\nDaily sales info:")
print(daily_sales.info())

assert not daily_sales.empty, "Daily sales dataset is empty!"
print("\nDataLoader is working correctly")
