# check_data_structure.py
import pandas as pd
from pathlib import Path
import sys

# Your actual data path
data_path = Path("D:/Inventory_Management/data/Inventory_Management")
print(f"📂 Checking data directory: {data_path}")

if not data_path.exists():
    print(f"❌ Directory does not exist!")
    sys.exit(1)

print(f"✅ Directory exists")

# List all files
print("\n📁 Files found:")
for file in sorted(data_path.glob("*")):
    if file.is_file():
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name:30} ({size_mb:.2f} MB)")

# Check for CSV files and their structure
print("\n🔍 CSV Files structure (first 3 rows):")
csv_files = list(data_path.glob("*.csv"))
if not csv_files:
    print("  No CSV files found!")
    # Check for other file types
    for ext in ['.xlsx', '.xls', '.parquet', '.json']:
        files = list(data_path.glob(f"*{ext}"))
        if files:
            print(f"  Found {len(files)} {ext} files")

for csv_file in csv_files[:5]:  # Check first 5 CSV files
    try:
        print(f"\n  File: {csv_file.name}")
        df = pd.read_csv(csv_file, nrows=3)
        print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"    Columns: {list(df.columns)}")
        print(f"    First 3 rows:")
        print(df.head(3).to_string())
        
        # Check if this might be a mapping file
        if 'item_id' in df.columns or 'id' in df.columns:
            print(f"    ⭐ Contains item ID column!")
        if any(col in df.columns for col in ['name', 'title', 'product_name', 'real_name', 'description']):
            print(f"    ⭐ Contains name column!")
            
    except Exception as e:
        print(f"    ❌ Error reading: {e}")

# Specifically look for item mapping
print("\n🎯 Looking for item ID to name mapping...")
for file in data_path.glob("*"):
    if file.is_file() and file.suffix in ['.csv', '.xlsx', '.xls']:
        try:
            if file.suffix == '.csv':
                df = pd.read_csv(file, nrows=5)
            else:
                df = pd.read_excel(file, nrows=5)
            
            # Check if this looks like a mapping file
            id_cols = [col for col in df.columns if 'item' in str(col).lower() or 'id' == str(col).lower()]
            name_cols = [col for col in df.columns if any(word in str(col).lower() for word in 
                          ['name', 'title', 'product', 'description', 'real'])]
            
            if id_cols and name_cols:
                print(f"\n✅ POTENTIAL MAPPING FILE: {file.name}")
                print(f"   ID columns: {id_cols}")
                print(f"   Name columns: {name_cols}")
                print(f"   Sample data:")
                print(df[[id_cols[0], name_cols[0]]].head().to_string())
                
        except Exception as e:
            continue

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Directory: {data_path}")
print(f"Total files: {len(list(data_path.glob('*')))}")
print(f"CSV files: {len(csv_files)}")
