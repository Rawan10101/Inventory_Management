"""
============================================================================
DATA LOADING TEST - Test Only Data Loading
============================================================================
Simple test to verify that your CSV files load correctly
No ML models, no predictions - just data loading validation
============================================================================
"""

import pandas as pd
from datetime import datetime
from enhanced_data_loader_v2_1 import EnhancedDataLoader

# Update this path to match your data location
DATA_PATH = "D:/Inventory_Management/data/Inventory_Management"
print(f"\n🔍 Starting data loading test...")
print(f"📁 Data path: {DATA_PATH}\n")
print("="*70)

try:
    # ========================================================================
    # STEP 1: Initialize Data Loader
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: INITIALIZE DATA LOADER")
    print("="*70)
    
    loader = EnhancedDataLoader(DATA_PATH)
    print("✅ Data loader initialized successfully\n")
    
    # ========================================================================
    # STEP 2: Load Sales Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: LOAD SALES DATA")
    print("="*70)
    
    sales_data = loader.prepare_daily_sales()
    
    if sales_data.empty:
        print("❌ FAILED: Sales data is empty!")
        print("   Check if these files exist:")
        print("   - fct_orders.csv")
        print("   - fct_order_items.csv")
    else:
        print(f"\n✅ SUCCESS: Loaded {len(sales_data):,} sales records")
        
        # Display basic statistics
        print("\n📊 Sales Data Statistics:")
        print(f"   Date Range: {sales_data['date'].min()} to {sales_data['date'].max()}")
        
        date_range_days = (pd.to_datetime(sales_data['date'].max()) - 
                          pd.to_datetime(sales_data['date'].min())).days
        print(f"   Total Days: {date_range_days}")
        print(f"   Unique Items: {sales_data['item_id'].nunique():,}")
        print(f"   Unique Places: {sales_data['place_id'].nunique():,}")
        print(f"   Total Quantity Sold: {sales_data['quantity_sold'].sum():,.0f}")
        print(f"   Total Revenue: ${sales_data['revenue'].sum():,.2f}")
        
        # Show column names
        print(f"\n📋 Available Columns ({len(sales_data.columns)}):")
        for i, col in enumerate(sales_data.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Show sample data
        print("\n📄 Sample Data (first 5 rows):")
        print(sales_data.head().to_string())
    
    # ========================================================================
    # STEP 3: Load Inventory Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: LOAD INVENTORY DATA")
    print("="*70)
    
    inventory_data = loader.prepare_inventory_snapshot()
    
    if inventory_data.empty:
        print("❌ FAILED: Inventory data is empty!")
        print("   Check if dim_items.csv exists")
    else:
        print(f"\n✅ SUCCESS: Loaded {len(inventory_data):,} inventory items")
        
        # Display basic statistics
        print("\n📊 Inventory Data Statistics:")
        print(f"   Items in Stock: {(inventory_data['current_stock'] > 0).sum():,}")
        print(f"   Out of Stock: {(inventory_data['current_stock'] == 0).sum():,}")
        print(f"   Average Stock: {inventory_data['current_stock'].mean():.1f} units")
        print(f"   Total Stock Value: ${inventory_data['total_value'].sum():,.2f}")
        print(f"   Stock Availability: {(inventory_data['current_stock'] > 0).sum() / len(inventory_data) * 100:.1f}%")
        
        # Show column names
        print(f"\n📋 Available Columns ({len(inventory_data.columns)}):")
        for i, col in enumerate(inventory_data.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Show sample data
        print("\n📄 Sample Data (first 5 rows):")
        display_cols = ['item_id', 'title', 'current_stock', 'unit_cost', 'price']
        available_cols = [col for col in display_cols if col in inventory_data.columns]
        print(inventory_data[available_cols].head().to_string())
    
    # ========================================================================
    # STEP 4: Data Quality Validation
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: DATA QUALITY VALIDATION")
    print("="*70)
    
    validation_results = loader.validate_data_quality()
    
    print("\n✅ Data quality validation completed")
    
    # ========================================================================
    # STEP 5: Loading Summary
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: LOADING SUMMARY")
    print("="*70)
    
    summary = loader.get_data_loading_summary()
    
    if not summary.empty:
        print("\n📋 Files Loaded:")
        print(summary.to_string(index=False))
        
        total_files = len(summary)
        loaded_files = len(summary[summary['loaded'] == '✅'])
        
        print(f"\n📊 Summary:")
        print(f"   Total Files: {total_files}")
        print(f"   Successfully Loaded: {loaded_files}")
        print(f"   Failed: {total_files - loaded_files}")
        print(f"   Success Rate: {loaded_files/total_files*100:.1f}%")
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("📊 FINAL TEST RESULTS")
    print("="*70 + "\n")
    
    # Check critical requirements
    tests_passed = 0
    tests_total = 3
    
    print("Critical Requirements:")
    
    # Test 1: Sales data exists
    if not sales_data.empty:
        print("✅ 1. Sales data loaded successfully")
        tests_passed += 1
    else:
        print("❌ 1. Sales data loading FAILED")
    
    # Test 2: Inventory data exists
    if not inventory_data.empty:
        print("✅ 2. Inventory data loaded successfully")
        tests_passed += 1
    else:
        print("❌ 2. Inventory data loading FAILED")
    
    # Test 3: Sufficient data for ML
    if not sales_data.empty and date_range_days >= 30:
        print(f"✅ 3. Sufficient data range ({date_range_days} days)")
        tests_passed += 1
    elif not sales_data.empty:
        print(f"⚠️  3. Limited data range ({date_range_days} days - recommended: 90+)")
    else:
        print("❌ 3. Insufficient data")
    
    print(f"\n📈 Test Score: {tests_passed}/{tests_total} ({tests_passed/tests_total*100:.0f}%)")
    
    if tests_passed == tests_total:
        print("\n🎉 PERFECT! All data loaded successfully!")
        print("✅ You can proceed to test the ML models")
    elif tests_passed >= 2:
        print("\n✅ GOOD! Critical data loaded successfully")
        print("⚠️  Some warnings but you can proceed")
    else:
        print("\n❌ FAILED: Critical data missing")
        print("⚠️  Fix data loading issues before testing ML models")
    
    print("\n" + "="*70)
    
    # Optional: Save results for inspection
    if not sales_data.empty:
        print("\n💾 Saving sample data for inspection...")
        sales_data.head(100).to_csv("sample_sales_data.csv", index=False)
        print("   ✅ Saved: sample_sales_data.csv")
    
    if not inventory_data.empty:
        inventory_data.head(100).to_csv("sample_inventory_data.csv", index=False)
        print("   ✅ Saved: sample_inventory_data.csv")
    
    print("\n✅ Data loading test completed!")
    print("="*70 + "\n")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: File not found")
    print(f"   {e}")
    print("\n💡 Make sure your data path is correct:")
    print(f"   Current: {DATA_PATH}")
    print("\n   Update DATA_PATH at the top of this script if needed")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    print("\n📋 Full error details:")
    print(traceback.format_exc())

print("\n" + "="*70)
print("END OF TEST")
print("="*70 + "\n")