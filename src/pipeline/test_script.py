# test_script.py
import sys
import os
from pathlib import Path

# Add parent directories to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Go up to Inventory_Management
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(ROOT_DIR / "src" / "pipeline"))

print(f"🔍 Looking for pipeline module...")
print(f"Current directory: {os.getcwd()}")
print(f"ROOT_DIR: {ROOT_DIR}")

# List Python files in pipeline directory
pipeline_dir = ROOT_DIR / "src" / "pipeline"
if pipeline_dir.exists():
    print(f"\n📁 Files in pipeline directory:")
    for file in pipeline_dir.glob("*.py"):
        print(f"  - {file.name}")

# Try to import
try:
    # First try importing pipeline_runner directly
    from pipeline_runner import run_pipeline
    print(f"\n✅ Successfully imported from pipeline_runner")
    
    # Run the pipeline
    print("\n🚀 Testing pipeline with real product names...")
    
    # Use relative paths from ROOT_DIR
    data_dir = str(ROOT_DIR / "data" / "Inventory_Management")
    output_dir = str(ROOT_DIR / "reports" / "test_run")
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Output directory: {output_dir}")
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Looking for: {list(Path(data_dir).parent.glob('*'))}")
    else:
        print(f"✅ Data directory exists")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        forecast_horizon="daily",
        prefer_advanced=True,
        prep_horizon_days=1
    )
    
    print("\n✅ Pipeline test completed!")
    print(f"📊 Results keys: {list(result.keys())}")
    
    if "demand_forecast" in result:
        forecast = result["demand_forecast"]
        print(f"📈 Forecast shape: {forecast.shape}")
        print(f"📈 Columns: {list(forecast.columns)}")
        
        if not forecast.empty and 'item_real_name' in forecast.columns:
            print(f"\n🏆 TOP 10 PREDICTIONS (with real names):")
            top_10 = forecast.nlargest(10, 'predicted_daily_demand')
            for idx, row in top_10.iterrows():
                name = row['item_real_name']
                demand = row.get('predicted_daily_demand', 0)
                conf = row.get('confidence', 0)
                print(f"  {name}: {demand:.1f} units (conf: {conf:.1%})")
        
        # Check if CSV files were created
        output_path = Path(output_dir)
        csv_files = list(output_path.glob("*.csv"))
        print(f"\n📄 CSV files created in {output_dir}:")
        for file in csv_files:
            print(f"  - {file.name} ({file.stat().st_size:,} bytes)")
            
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    
    # Try alternative imports
    print("\n🔧 Trying alternative imports...")
    try:
        import importlib.util
        
        # Try to load module directly
        module_path = pipeline_dir / "pipeline_runner.py"
        if module_path.exists():
            print(f"Found module at: {module_path}")
            
            spec = importlib.util.spec_from_file_location("pipeline_runner", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'run_pipeline'):
                print("✅ Successfully loaded module directly")
                run_pipeline = module.run_pipeline
                
                # Now run the pipeline
                print("\n🚀 Testing pipeline...")
                data_dir = str(ROOT_DIR / "data" / "Inventory_Management")
                output_dir = str(ROOT_DIR / "reports" / "test_run")
                
                result = run_pipeline(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    forecast_horizon="daily",
                    prefer_advanced=True
                )
                print("✅ Pipeline ran successfully!")
            else:
                print("❌ Module doesn't have run_pipeline function")
        else:
            print(f"❌ Module not found at: {module_path}")
            
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"\n❌ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()