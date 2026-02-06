"""
============================================================================
ENHANCED DATA LOADER - COMPETITION EDITION
============================================================================
Comprehensive data loading with validation and intelligent fallbacks
Supports all business requirements including BOM, campaigns, and pricing
============================================================================
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class EnhancedDataLoader:
    """
    Enhanced Data Loader for Fresh Flow Competition
    
    Features:
    - Comprehensive dataset loading (15+ tables)
    - Intelligent mock data generation
    - Data validation and quality checks
    - Automatic date conversion (UNIX timestamps)
    - Active merchant filtering
    - Price and cost estimation
    - Shelf life intelligent defaults
    """
    
    def __init__(self, data_path: str):
        """
        Initialize Enhanced Data Loader
        
        Args:
            data_path: Path to data directory containing CSV files
        """
        self.data_path = data_path
        self.data = None
        self.sales_data = None
        
        print(f"📁 Enhanced Data Loader Initialized")
        print(f"   Data Path: {data_path}\n")
    
    # ========================================================================
    # CORE LOADING METHODS
    # ========================================================================
    
    def load_csv(self, filename: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load CSV with error handling and logging
        
        Args:
            filename: CSV filename
            parse_dates: Columns to parse as dates
        
        Returns:
            Loaded DataFrame
        """
        file_path = f"{self.data_path}/{filename}"
        try:
            df = pd.read_csv(file_path, parse_dates=parse_dates, low_memory=False)
            df.columns = df.columns.str.strip()  # Clean column names
            print(f"✅ Loaded {filename}: {len(df):,} rows, {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            print(f"⚠️  File not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def convert_unix_timestamp(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Convert UNIX timestamp to datetime
        
        Args:
            df: DataFrame
            column: Column name containing UNIX timestamp
        
        Returns:
            DataFrame with converted timestamp
        """
        if column in df.columns:
            df[f'{column}_datetime'] = pd.to_datetime(df[column], unit='s', errors='coerce')
            print(f"   Converted {column} to datetime")
        return df
    
    # ========================================================================
    # FACT TABLE LOADERS
    # ========================================================================
    
    def load_orders(self) -> pd.DataFrame:
        """
        Load fct_orders with timestamp conversion
        
        Returns:
            Orders DataFrame with datetime conversion
        """
        print("\n📦 Loading Orders...")
        orders = self.load_csv("fct_orders.csv")
        
        if orders.empty:
            return orders
        
        # Convert timestamps
        orders = self.convert_unix_timestamp(orders, 'created')
        
        # Keep only closed orders
        if 'status' in orders.columns:
            original_count = len(orders)
            orders = orders[orders['status'] == 'Closed']
            print(f"   Filtered to closed orders: {len(orders):,} / {original_count:,}")
        
        # Rename for clarity
        if 'created_datetime' in orders.columns:
            orders = orders.rename(columns={'created_datetime': 'order_created_at'})
        
        return orders
    
    def load_order_items(self) -> pd.DataFrame:
        """Load fct_order_items - individual items in orders"""
        print("\n🛒 Loading Order Items...")
        return self.load_csv("fct_order_items.csv")
    
    def load_inventory_reports(self) -> pd.DataFrame:
        """
        Load fct_inventory_reports with date conversion
        
        Returns:
            Inventory reports or empty DataFrame
        """
        print("\n📊 Loading Inventory Reports...")
        df = self.load_csv("fct_inventory_reports.csv")
        
        if not df.empty and 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        
        return df
    
    def load_campaigns(self) -> pd.DataFrame:
        """
        Load fct_campaigns - marketing campaign execution records
        
        Returns:
            Campaigns DataFrame
        """
        print("\n📢 Loading Campaigns...")
        campaigns = self.load_csv("fct_campaigns.csv")
        
        if not campaigns.empty:
            # Convert dates
            for col in ['start_date', 'end_date', 'created']:
                campaigns = self.convert_unix_timestamp(campaigns, col)
        
        return campaigns
    
    def load_bonus_codes(self) -> pd.DataFrame:
        """
        Load fct_bonus_codes - promotional codes
        
        Returns:
            Bonus codes DataFrame
        """
        print("\n🎟️  Loading Bonus Codes...")
        codes = self.load_csv("fct_bonus_codes.csv")
        
        if not codes.empty:
            # Convert dates
            for col in ['valid_from', 'valid_to', 'created']:
                codes = self.convert_unix_timestamp(codes, col)
        
        return codes
    
    # ========================================================================
    # DIMENSION TABLE LOADERS
    # ========================================================================
    
    def load_items(self) -> pd.DataFrame:
        """
        Load dim_items - raw inventory items catalog
        
        Enhanced with intelligent defaults for missing data
        """
        print("\n📦 Loading Items...")
        items = self.load_csv("dim_items.csv")
        
        if items.empty:
            return items
        
        # Add shelf life if missing
        if 'shelf_life_days' not in items.columns:
            items['shelf_life_days'] = items.apply(self._estimate_shelf_life, axis=1)
            print("   Added intelligent shelf life estimates")
        
        # Ensure manage_inventory exists
        if 'manage_inventory' not in items.columns:
            items['manage_inventory'] = 1  # Default to managed
            print("   Added manage_inventory flag (default=1)")
        
        return items
    
    def load_menu_items(self) -> pd.DataFrame:
        """Load dim_menu_items - merchant's product setup"""
        print("\n🍽️  Loading Menu Items...")
        return self.load_csv("dim_menu_items.csv")
    
    def load_bill_of_materials(self) -> pd.DataFrame:
        """
        Load dim_bill_of_materials - recipe ingredient breakdown
        
        Critical for kitchen prep calculator
        """
        print("\n🧪 Loading Bill of Materials...")
        bom = self.load_csv("dim_bill_of_materials.csv")
        
        if bom.empty:
            print("   ⚠️  BOM not available - will affect prep calculator")
            return bom
        
        # Validate required columns
        required_cols = ['menu_item_id', 'raw_item_id', 'quantity']
        missing_cols = [col for col in required_cols if col not in bom.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
        else:
            print(f"   ✅ BOM validated: {len(bom)} recipes")
        
        # Add unit if missing
        if 'unit' not in bom.columns:
            bom['unit'] = 'units'
        
        return bom
    
    def load_places(self) -> pd.DataFrame:
        """
        Load dim_places - merchant/shop information
        
        Returns:
            Places DataFrame with date conversion
        """
        print("\n🏪 Loading Places...")
        places = self.load_csv("dim_places.csv")
        
        if not places.empty:
            # Convert dates
            for col in ['contract_start', 'termination_date', 'created']:
                places = self.convert_unix_timestamp(places, col)
        
        return places
    
    def load_add_ons(self) -> pd.DataFrame:
        """Load dim_add_ons - individual add-on options"""
        print("\n➕ Loading Add-Ons...")
        return self.load_csv("dim_add_ons.csv")
    
    def load_menu_item_add_ons(self) -> pd.DataFrame:
        """Load dim_menu_item_add_ons - links add-ons to menu items"""
        print("\n🔗 Loading Menu Item Add-Ons...")
        return self.load_csv("dim_menu_item_add_ons.csv")
    
    def load_stock_categories(self) -> pd.DataFrame:
        """Load dim_stock_categories - inventory classification"""
        print("\n📑 Loading Stock Categories...")
        return self.load_csv("dim_stock_categories.csv")
    
    def load_users(self) -> pd.DataFrame:
        """Load dim_users - internal staff, merchant staff, consumers"""
        print("\n👥 Loading Users...")
        users = self.load_csv("dim_users.csv")
        
        if not users.empty and 'created' in users.columns:
            users = self.convert_unix_timestamp(users, 'created')
        
        return users
    
    def debug_data_sources(self):
        """
        Print a summary of which data sources are available and being used
        """
        print("\n📌 DATA SOURCE SUMMARY")
        print("=" * 40)

        # Sales data (used for forecasting)
        sales_rows = 0 if self.sales_data is None else len(self.sales_data)
        print(f"Sales data rows loaded: {sales_rows:,}")

        # Inventory reports (real vs mock)
        inventory_reports = self.load_inventory_reports()
        print(f"Inventory reports available: {not inventory_reports.empty}")

        # Bill of Materials (prep calculator)
        bom = self.load_bill_of_materials()
        print(f"BOM available: {not bom.empty}")

        print("=" * 40 + "\n")

    def load_taxonomy_terms(self) -> pd.DataFrame:
        """Load dim_taxonomy_terms - standardized lists and categories"""
        print("\n🏷️  Loading Taxonomy Terms...")
        return self.load_csv("dim_taxonomy_terms.csv")
    
    # ========================================================================
    # AGGREGATED VIEW LOADERS
    # ========================================================================
    
    def load_most_ordered(self) -> pd.DataFrame:
        """Load most_ordered - pre-aggregated top-selling items"""
        print("\n📈 Loading Most Ordered Items...")
        return self.load_csv("most_ordered.csv")
    
    # ========================================================================
    # DATA PREPARATION METHODS
    # ========================================================================
    
    def filter_active_merchants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only active merchants (no termination date)
        
        Args:
            df: DataFrame with termination_date column
        
        Returns:
            Filtered DataFrame
        """
        if 'termination_date' not in df.columns:
            return df
        
        # Check both termination_date and termination_date_datetime
        date_col = 'termination_date_datetime' if 'termination_date_datetime' in df.columns else 'termination_date'
        
        original_count = len(df)
        active_df = df[df[date_col].isna()]
        
        print(f"   Active merchants: {len(active_df):,} / {original_count:,}")
        return active_df
    
    def merge_datasets(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                      left_on: str, right_on: str, how: str = "inner") -> pd.DataFrame:
        """
        Merge two datasets with logging
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            left_on: Left join key
            right_on: Right join key
            how: Join type
        
        Returns:
            Merged DataFrame
        """
        merged = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
        print(f"   Merged: {len(left_df):,} + {len(right_df):,} → {len(merged):,} rows")
        return merged
    
    def prepare_daily_sales(self) -> pd.DataFrame:
        """
        Prepare daily sales data for demand forecasting
        
        Aggregates order items to daily level with metadata
        
        Returns:
            Daily sales DataFrame ready for ML
        """
        print("\n" + "="*70)
        print("📊 PREPARING DAILY SALES DATA")
        print("="*70)
        
        orders = self.load_orders()
        order_items = self.load_order_items()
        
        if orders.empty or order_items.empty:
            print("❌ Cannot prepare sales data - missing orders or order items")
            return pd.DataFrame()
        
        # Merge order-level info into items
        df = self.merge_datasets(
            order_items,
            orders[["id", "place_id", "order_created_at"]],
            left_on="order_id",
            right_on="id"
        )
        
        # Ensure datetime
        df["order_created_at"] = pd.to_datetime(df["order_created_at"], errors="coerce")
        
        # Extract date
        df["date"] = df["order_created_at"].dt.date

                # Ensure cost exists
        if "cost" in df.columns:
            df["item_cost"] = df["cost"]
        else:
            df["item_cost"] = df["price"] * 0.6  # fallback

        
        # Aggregate to daily level
        print("\n   Aggregating to daily level...")
       
    

        group_cols = ["place_id", "item_id", "date"]

        for col in ["channel", "type", "platform"]:
            if col in df.columns:
                group_cols.append(col)

        daily_sales = (
        df.groupby(group_cols)
        .agg(
            quantity_sold=("quantity", "sum"),
            revenue=("price", "sum"),
            cost=("item_cost", "sum"),
            num_orders=("order_id", "nunique")
        )
        .reset_index()
    )


        # === Campaign impact flag ===
        campaigns = self.load_campaigns()

        daily_sales["is_campaign_active"] = 0

        if not campaigns.empty:
            if "start_date_datetime" in campaigns.columns and "end_date_datetime" in campaigns.columns:
                campaigns["start_date"] = campaigns["start_date_datetime"].dt.date
                campaigns["end_date"] = campaigns["end_date_datetime"].dt.date

                for _, c in campaigns.iterrows():
                    daily_sales.loc[
                        (daily_sales["date"] >= c["start_date"]) &
                        (daily_sales["date"] <= c["end_date"]),
                        "is_campaign_active"
                    ] = 1

                print("   Added campaign activity flag")

        
        # Merge item metadata
        items = self.load_items()
        if not items.empty:
            daily_sales = daily_sales.merge(
                items[["id", "title", "type", "manage_inventory"]],
                left_on="item_id",
                right_on="id",
                how="left"
            )
            print(f"   Added item metadata")
        
        # Merge place metadata
        places = self.load_places()
        if not places.empty:
            active_places = self.filter_active_merchants(places)
            daily_sales = daily_sales.merge(
                active_places[["id", "title"]],
                left_on="place_id",
                right_on="id",
                how="left",
                suffixes=("", "_place")
            )
            print(f"   Added place metadata")
        
        # Store for inventory snapshot
        self.sales_data = daily_sales
        
        print(f"\n✅ Daily Sales Prepared: {len(daily_sales):,} records")
        print(f"   Date Range: {daily_sales['date'].min()} to {daily_sales['date'].max()}")
        print(f"   Unique Items: {daily_sales['item_id'].nunique():,}")
        print(f"   Unique Places: {daily_sales['place_id'].nunique():,}")
        print("="*70 + "\n")
        
        return daily_sales
    
    def prepare_inventory_snapshot(self) -> pd.DataFrame:
        """
        Prepare current inventory snapshot
        
        Uses actual inventory reports if available, otherwise creates
        intelligent mock data based on sales patterns
        
        Returns:
            Inventory snapshot DataFrame
        """
        print("\n" + "="*70)
        print("📦 PREPARING INVENTORY SNAPSHOT")
        print("="*70)
        
        inventory_reports = self.load_inventory_reports()
        items = self.load_items()
        
        if items.empty:
            print("❌ Cannot prepare inventory - items table missing")
            return pd.DataFrame()
        
        # Keep only inventory-managed items
        inventory_items = items[items["manage_inventory"] == 1].copy()
        print(f"\n   Inventory-managed items: {len(inventory_items):,}")
        
        # ===== STRATEGY 1: Use actual inventory reports =====
        if not inventory_reports.empty and 'item_id' in inventory_reports.columns:
            print("   ✅ Using actual inventory reports")
            
            # Get most recent report
            if 'report_date' in inventory_reports.columns:
                latest_date = inventory_reports['report_date'].max()
                inventory_reports = inventory_reports[
                    inventory_reports['report_date'] == latest_date
                ]
                print(f"   Latest report date: {latest_date}")
            
            inventory = inventory_reports.merge(
                inventory_items[["id", "title", "type", "shelf_life_days"]],
                left_on="item_id",
                right_on="id",
                how="inner"
            )
            
            inventory = inventory.rename(columns={"quantity_on_hand": "current_stock"})
            
        # ===== STRATEGY 2: Create intelligent mock inventory =====
        else:
            print("   ⚠️  No inventory reports - generating intelligent mock data")
            
            inventory = inventory_items.copy()
            inventory = inventory.rename(columns={"id": "item_id"})
            
            # Use sales data to estimate realistic stock levels
            if self.sales_data is not None and len(self.sales_data) > 0:
                print("   Using sales patterns for stock estimation...")
                
                # Calculate sales statistics
                sales_stats = self.sales_data.groupby('item_id').agg({
                    'quantity_sold': ['mean', 'std', 'max', 'sum'],
                    'date': 'count'
                }).reset_index()
                
                sales_stats.columns = ['item_id', 'avg_daily_sales', 'std_daily_sales', 
                                      'max_daily_sales', 'total_sales', 'days_sold']
                
                # Merge with inventory
                inventory = inventory.merge(sales_stats, on='item_id', how='left')
                
                # Estimate stock levels intelligently
                # Stock = (7-14 days of avg sales) × random factor
                np.random.seed(42)
                inventory['days_of_stock'] = np.random.uniform(7, 14, len(inventory))
                
                inventory['current_stock'] = (
                    inventory['avg_daily_sales'].fillna(5) * inventory['days_of_stock']
                ).round(0).astype(int)
                
                # Add realistic variation (some overstocked, some understocked)
                random_factor = np.random.uniform(0.6, 1.4, len(inventory))
                inventory['current_stock'] = (
                    inventory['current_stock'] * random_factor
                ).round(0).astype(int)
                
                # Ensure minimum of 0
                inventory['current_stock'] = inventory['current_stock'].clip(lower=0)
                
                # Some items out of stock (realistic)
                out_of_stock_pct = 0.15  # 15% out of stock
                out_of_stock_mask = np.random.random(len(inventory)) < out_of_stock_pct
                inventory.loc[out_of_stock_mask, 'current_stock'] = 0
                
                # Fast movers should have higher stock
                inventory.loc[inventory['avg_daily_sales'] > inventory['avg_daily_sales'].quantile(0.75), 
                             'current_stock'] *= 1.5
                inventory['current_stock'] = inventory['current_stock'].round(0).astype(int)
                
                print(f"   Generated stock based on {len(sales_stats)} items with sales history")
                
            else:
                # Fallback: simple defaults
                print("   Using default stock levels (no sales data)")
                inventory['current_stock'] = np.random.randint(0, 100, len(inventory))
                
                # Inventory risk indicators
        if "variance" in inventory.columns:
            inventory["stock_variance_rate"] = (
                inventory["variance"] / inventory["current_stock"]
            ).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            inventory["stock_variance_rate"] = 0

        # ===== ENRICH INVENTORY DATA =====
        
        # Days in stock (random 0-10 days)
        if 'days_in_stock' not in inventory.columns:
            inventory['days_in_stock'] = np.random.randint(0, 10, len(inventory))
        
        # Unit cost estimation
        if 'unit_cost' not in inventory.columns:
            # Try to get from price (assume 40% margin → cost = 60% of price)
            if 'price' in inventory.columns:
                inventory['unit_cost'] = inventory['price'] * 0.6
            else:
                # Estimate from sales data
                if self.sales_data is not None and 'revenue' in self.sales_data.columns:
                    avg_prices = (
                        self.sales_data.groupby('item_id')['revenue'].sum() / 
                        self.sales_data.groupby('item_id')['quantity_sold'].sum()
                    ).reset_index()
                    avg_prices.columns = ['item_id', 'estimated_price']
                    
                    inventory = inventory.merge(avg_prices, on='item_id', how='left')
                    inventory['unit_cost'] = inventory['estimated_price'].fillna(30) * 0.6
                    inventory['price'] = inventory['estimated_price'].fillna(50)
                else:
                    inventory['unit_cost'] = 30  # Default
                    inventory['price'] = 50
        
        # Ensure price exists
        if 'price' not in inventory.columns:
            if 'unit_cost' in inventory.columns:
                inventory['price'] = inventory['unit_cost'] / 0.6  # Reverse calculate
            else:
                inventory['price'] = 50  # Default
        
        # Calculate total value
        inventory['total_value'] = inventory['current_stock'] * inventory['unit_cost']
        
        # Add reorder thresholds if missing
        if 'reorder_point' not in inventory.columns:
            # Reorder point = 3 days of average sales
            if 'avg_daily_sales' in inventory.columns:
                inventory['reorder_point'] = (inventory['avg_daily_sales'] * 3).fillna(10)
            else:
                inventory['reorder_point'] = 10  # Default
        
        if 'reorder_quantity' not in inventory.columns:
            # Reorder quantity = 7 days of average sales
            if 'avg_daily_sales' in inventory.columns:
                inventory['reorder_quantity'] = (inventory['avg_daily_sales'] * 7).fillna(50)
            else:
                inventory['reorder_quantity'] = 50  # Default
        
        # ===== SUMMARY STATISTICS =====
        print(f"\n📊 Inventory Snapshot Summary:")
        print(f"   Total Items: {len(inventory):,}")
        print(f"   Items in Stock: {(inventory['current_stock'] > 0).sum():,}")
        print(f"   Out of Stock: {(inventory['current_stock'] == 0).sum():,}")
        print(f"   Average Stock Level: {inventory['current_stock'].mean():.1f} units")
        print(f"   Total Inventory Value: {inventory['total_value'].sum():,.2f} DKK")
        print(f"   Stock Availability: {(inventory['current_stock'] > 0).sum() / len(inventory) * 100:.1f}%")
        print("="*70 + "\n")
        
        return inventory
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _estimate_shelf_life(self, row) -> int:
        """
        Intelligently estimate shelf life based on item characteristics
        
        Args:
            row: DataFrame row with item info
        
        Returns:
            Estimated shelf life in days
        """
        item_type = str(row.get('type', '')).lower()
        title = str(row.get('title', '')).lower()
        
        # Highly perishable (dairy, produce, prepared)
        if any(word in title for word in [
            'milk', 'cream', 'yogurt', 'cheese', 'salad', 
            'sandwich', 'fresh', 'juice', 'smoothie'
        ]):
            return 3
        
        # Perishable (baked goods, coffee drinks)
        if any(word in title for word in [
            'bread', 'pastry', 'cake', 'muffin', 'croissant',
            'coffee', 'latte', 'cappuccino', 'espresso'
        ]):
            return 7
        
        # Semi-perishable (packaged foods)
        if any(word in item_type for word in ['packaged', 'canned', 'bottled']):
            return 30
        
        # Long shelf life (dry goods, frozen)
        if any(word in item_type for word in ['frozen', 'dry', 'grain', 'pasta', 'rice']):
            return 90
        
        # Non-perishable
        if any(word in item_type for word in ['beverage', 'snack', 'condiment']):
            return 60
        
        # Default
        return 14
    
    
    
    def validate_data_quality(self) -> Dict:
        """
        Run data quality checks
        
        Returns:
            Dictionary with validation results
        """
        print("\n" + "="*70)
        print("🔍 DATA QUALITY VALIDATION")
        print("="*70 + "\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check 1: Sales data coverage
        if self.sales_data is not None:
            date_range = (
                self.sales_data['date'].max() - self.sales_data['date'].min()
            ).days
            results['checks'].append({
                'check': 'Sales Data Coverage',
                'status': 'PASS' if date_range >= 90 else 'WARNING',
                'details': f'{date_range} days of history'
            })
        
        # Check 2: Missing values
        if self.sales_data is not None:
            missing_pct = (self.sales_data.isnull().sum() / len(self.sales_data) * 100).max()
            results['checks'].append({
                'check': 'Missing Values',
                'status': 'PASS' if missing_pct < 10 else 'WARNING',
                'details': f'{missing_pct:.1f}% max missing'
            })
        
        # Print results
        for check in results['checks']:
            status_icon = '✅' if check['status'] == 'PASS' else '⚠️'
            print(f"{status_icon} {check['check']}: {check['details']}")
        
        print("\n" + "="*70 + "\n")
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║              ENHANCED DATA LOADER - COMPETITION EDITION                ║
    ║                                                                        ║
    ║  Comprehensive data loading for Fresh Flow Intelligence System         ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example usage
    # loader = EnhancedDataLoader("path/to/data")
    # sales = loader.prepare_daily_sales()
    # inventory = loader.prepare_inventory_snapshot()
    # bom = loader.load_bill_of_materials()