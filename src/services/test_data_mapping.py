#!/usr/bin/env python3
"""
Test script to check data mapping issues
"""

import pandas as pd
import os
from enhanced_data_loader_v2_1 import EnhancedDataLoader

# Initialize
data_path = "D:/Inventory_Management/data/Inventory_Management"
loader = EnhancedDataLoader(data_path)

print("="*80)
print("🧪 TESTING DATA MAPPING ISSUES")
print("="*80)

# 1. Load all tables and check their structure
print("\n📊 1. CHECKING DIM_ITEMS (Item Catalog):")
items_df = loader.load_items()
print(f"   Shape: {items_df.shape}")
print(f"   Columns: {list(items_df.columns)}")
print(f"   First few rows:")
print(items_df[['id', 'title', 'type', 'manage_inventory']].head(10).to_string())

# 2. Check BOM structure
print("\n🧪 2. CHECKING BILL OF MATERIALS:")
bom_df = loader.load_bill_of_materials()
print(f"   Shape: {bom_df.shape}")
print(f"   Columns: {list(bom_df.columns)}")
if not bom_df.empty:
    print(f"   First few rows:")
    print(bom_df.head(10).to_string())
else:
    print("   ⚠️  BOM is empty or failed to load!")

# 3. Check menu items
print("\n🍽️  3. CHECKING MENU ITEMS:")
menu_items_df = loader.load_menu_items()
print(f"   Shape: {menu_items_df.shape}")
if not menu_items_df.empty:
    print(f"   Columns: {list(menu_items_df.columns)}")
    print(f"   First few rows:")
    print(menu_items_df.head(10).to_string())

# 4. Prepare sales data and check mapping
print("\n💰 4. CHECKING SALES DATA MAPPING:")
sales_df = loader.prepare_daily_sales()
if sales_df is not None and not sales_df.empty:
    print(f"   Shape: {sales_df.shape}")
    print(f"   Columns with 'title': {[col for col in sales_df.columns if 'title' in col]}")
    
    # Check if item names are in sales data
    if 'title' in sales_df.columns:
        print(f"   Item names in sales data: {sales_df['title'].nunique()} unique names")
        print(f"   Sample mapping:")
        sample = sales_df[['item_id', 'title']].drop_duplicates().head(10)
        print(sample.to_string())
    else:
        print("   ❌ No 'title' column in sales data!")
    
    # Check item_id range
    print(f"\n   Item ID statistics:")
    print(f"   Unique item IDs: {sales_df['item_id'].nunique()}")
    print(f"   Min item ID: {sales_df['item_id'].min()}")
    print(f"   Max item ID: {sales_df['item_id'].max()}")
    print(f"   IDs not in dim_items: {set(sales_df['item_id']) - set(items_df['id'])}")

# 5. Check inventory snapshot
print("\n📦 5. CHECKING INVENTORY SNAPSHOT:")
inventory_df = loader.prepare_inventory_snapshot()
if inventory_df is not None and not inventory_df.empty:
    print(f"   Shape: {inventory_df.shape}")
    print(f"   Columns: {list(inventory_df.columns)}")
    
    if 'title' in inventory_df.columns:
        print(f"\n   Inventory item names:")
        sample = inventory_df[['item_id', 'title', 'current_stock']].head(10)
        print(sample.to_string())
    else:
        print("   ❌ No 'title' column in inventory!")

# 6. Check key relationships
print("\n🔗 6. CHECKING KEY RELATIONSHIPS:")
if not items_df.empty and not sales_df.empty:
    print(f"   Items in catalog: {len(items_df)}")
    print(f"   Items with sales: {sales_df['item_id'].nunique()}")
    
    # Check for orphaned items (in catalog but no sales)
    catalog_ids = set(items_df['id'])
    sales_ids = set(sales_df['item_id'])
    orphaned = catalog_ids - sales_ids
    print(f"   Items in catalog but no sales: {len(orphaned)}")
    if orphaned:
        print(f"   Orphaned item IDs: {sorted(list(orphaned))[:10]}...")

# 7. Debug BOM column issue
print("\n🔍 7. DEBUGGING BOM COLUMN NAMES:")
if not bom_df.empty:
    print(f"   BOM actual columns: {list(bom_df.columns)}")
    print(f"   Looking for menu item column...")
    
    # Common column names for menu items in BOM
    possible_menu_cols = ['menu_item_id', 'menuitem_id', 'menu_item', 'menuitem', 'id', 'menu_id']
    found_menu_col = None
    for col in possible_menu_cols:
        if col in bom_df.columns:
            found_menu_col = col
            break
    
    if found_menu_col:
        print(f"   ✅ Found menu item column: '{found_menu_col}'")
        print(f"   Unique menu items in BOM: {bom_df[found_menu_col].nunique()}")
    else:
        print(f"   ❌ No standard menu item column found in BOM!")
        print(f"   Available columns: {list(bom_df.columns)}")

print("\n" + "="*80)
print("✅ TEST COMPLETE")
print("="*80)