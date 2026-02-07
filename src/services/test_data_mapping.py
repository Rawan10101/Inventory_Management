#!/usr/bin/env python3
"""
Test only the item mapping functionality
"""

from enhanced_data_loader_v2_1 import EnhancedDataLoader

print("="*80)
print(" TESTING ITEM MAPPING ONLY")
print("="*80)

# Initialize
data_path = "D:/Inventory_Management/data/Inventory_Management"
loader = EnhancedDataLoader(data_path)

# Test 1: Create mapping
print("\n1. CREATING ITEM MAPPING:")
item_mapping = loader.create_item_mapping()

if not item_mapping:
    print("   ❌ Failed to create mapping!")
    exit(1)

print(f"   ✅ Created mapping for {len(item_mapping)} items")

# Test 2: Get real names
print("\n2. GETTING REAL NAMES:")
for item_id in range(1, 31):  # Items 1-30 (the ones with sales)
    real_name = loader.get_real_item_name(item_id)
    print(f"   Item {item_id:2d}: {real_name}")

# Test 3: Check mapping details
print("\n3. MAPPING DETAILS (first 5 items):")
for item_id in range(1, 6):
    if item_id in item_mapping:
        mapping = item_mapping[item_id]
        print(f"   Item {item_id}:")
        print(f"     Simple: {mapping['simple_name']}")
        print(f"     Real:   {mapping['real_name']}")
        print(f"     Price:  ${mapping['real_price']:.2f}")
        print(f"     Type:   {mapping['mapping_type']}")

print("\n" + "="*80)
print("✅ MAPPING TEST COMPLETE")
print("="*80)