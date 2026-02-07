"""
============================================================================
ENHANCED DATA LOADER - UPDATED WITH PURCHASE-SALES INVENTORY METHOD
============================================================================
Comprehensive data loading with REAL inventory calculation
NEW: Calculates inventory from Invoice Purchases - Order Sales
Version: 2.1 - Updated Inventory Strategy
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
    Enhanced Data Loader for Fresh Flow Competition - UPDATED VERSION
    
    NEW FEATURES:
    - Real inventory calculation: Purchases - Sales = Current Stock
    - Uses fct_invoice_items for actual purchases
    - Uses fct_order_items for actual sales
    - Automatic cost tracking from invoices
    - Multiple fallback strategies
    
    All other features remain:
    - Comprehensive dataset loading (20+ tables)
    - Intelligent mock data generation
    - Data validation and quality checks
    - Automatic date conversion (UNIX timestamps)
    - Active merchant filtering
    - Campaign integration
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
        self.loaded_tables = {}
        
        print(f"📁 Enhanced Data Loader Initialized (v2.1 - Purchase-Sales Inventory)")
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
            self.loaded_tables[filename] = {
                'rows': len(df),
                'columns': len(df.columns),
                'loaded': True,
                'timestamp': datetime.now()
            }
            return df
        except FileNotFoundError:
            print(f"⚠️  File not found: {filename}")
            self.loaded_tables[filename] = {
                'rows': 0,
                'columns': 0,
                'loaded': False,
                'error': 'File not found'
            }
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            self.loaded_tables[filename] = {
                'rows': 0,
                'columns': 0,
                'loaded': False,
                'error': str(e)
            }
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
        """Load fct_orders with timestamp conversion"""
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
        """Load fct_inventory_reports with date conversion"""
        print("\n📊 Loading Inventory Reports...")
        df = self.load_csv("fct_inventory_reports.csv")
        
        if not df.empty and 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        
        return df
    
    def load_campaigns(self) -> pd.DataFrame:
        """Load fct_campaigns - marketing campaign execution records"""
        print("\n📢 Loading Campaigns...")
        campaigns = self.load_csv("fct_campaigns.csv")
        
        if not campaigns.empty:
            for col in ['start_date', 'end_date', 'created']:
                campaigns = self.convert_unix_timestamp(campaigns, col)
        
        return campaigns
    
    def load_bonus_codes(self) -> pd.DataFrame:
        """Load fct_bonus_codes - promotional codes"""
        print("\n🎟️  Loading Bonus Codes...")
        codes = self.load_csv("fct_bonus_codes.csv")
        
        if not codes.empty:
            for col in ['valid_from', 'valid_to', 'created']:
                codes = self.convert_unix_timestamp(codes, col)
        
        return codes
    
    def load_invoice_items(self) -> pd.DataFrame:
        """
        Load fct_invoice_items - supplier invoice line items
        
        CRITICAL: Used for actual cost tracking AND inventory calculation
        
        Returns:
            Invoice items DataFrame
        """
        print("\n📄 Loading Invoice Items...")
        invoices = self.load_csv("fct_invoice_items.csv")
        
        if not invoices.empty:
            # Calculate total amount if missing
            if 'total_amount' not in invoices.columns and all(col in invoices.columns for col in ['quantity', 'unit_price']):
                invoices['total_amount'] = invoices['quantity'] * invoices['unit_price']
                print("   Calculated total_amount from quantity × unit_price")
        
        return invoices
    
    def load_cash_balances(self) -> pd.DataFrame:
        """Load fct_cash_balances - daily cash reconciliation"""
        print("\n💰 Loading Cash Balances...")
        df = self.load_csv("fct_cash_balances.csv")
        
        if not df.empty and 'balance_date' in df.columns:
            df['balance_date'] = pd.to_datetime(df['balance_date'], errors='coerce')
            
            if 'variance' not in df.columns and all(col in df.columns for col in ['expected_cash', 'actual_cash']):
                df['variance'] = df['actual_cash'] - df['expected_cash']
                print("   Calculated variance from actual - expected cash")
        
        return df
    
    # ========================================================================
    # DIMENSION TABLE LOADERS
    # ========================================================================
    
    def load_items(self) -> pd.DataFrame:
        """Load dim_items - raw inventory items catalog"""
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
            items['manage_inventory'] = 1
            print("   Added manage_inventory flag (default=1)")
        
        return items
    
    def load_menu_items(self) -> pd.DataFrame:
        """Load dim_menu_items - merchant's product setup"""
        print("\n🍽️  Loading Menu Items...")
        return self.load_csv("dim_menu_items.csv")
    
    def load_bill_of_materials(self) -> pd.DataFrame:
        """Load dim_bill_of_materials - recipe ingredient breakdown"""
        print("\n🧪 Loading Bill of Materials...")
        bom = self.load_csv("dim_bill_of_materials.csv")
        
        if bom.empty:
            print("   ⚠️  BOM not available - will affect prep calculator")
            return bom
        
        required_cols = ['menu_item_id', 'raw_item_id', 'quantity']
        missing_cols = [col for col in required_cols if col not in bom.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
        else:
            print(f"   ✅ BOM validated: {len(bom)} recipes")
        
        if 'unit' not in bom.columns:
            bom['unit'] = 'units'
        
        return bom
    
    def load_places(self) -> pd.DataFrame:
        """Load dim_places - merchant/shop information"""
        print("\n🏪 Loading Places...")
        places = self.load_csv("dim_places.csv")
        
        if not places.empty:
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
        
        if not users.empty:
            if 'created' in users.columns:
                users = self.convert_unix_timestamp(users, 'created')
            
            if 'type' in users.columns:
                user_type_counts = users['type'].value_counts()
                print("   User type distribution:")
                for user_type, count in user_type_counts.items():
                    print(f"     {user_type}: {count:,}")
        
        return users
    
    def load_taxonomy_terms(self) -> pd.DataFrame:
        """Load dim_taxonomy_terms - standardized lists and categories"""
        print("\n🏷️  Loading Taxonomy Terms...")
        taxonomy = self.load_csv("dim_taxonomy_terms.csv")
        
        if not taxonomy.empty and 'vocabulary' in taxonomy.columns:
            vocab_counts = taxonomy['vocabulary'].value_counts()
            print("   Taxonomy vocabularies:")
            for vocab, count in vocab_counts.items():
                print(f"     {vocab}: {count:,} terms")
        
        return taxonomy
    
    def load_skus(self) -> pd.DataFrame:
        """Load dim_skus - product variants and SKU codes"""
        print("\n🏷️  Loading SKUs...")
        return self.load_csv("dim_skus.csv")
    
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
        """Filter to only active merchants (no termination date)"""
        if 'termination_date' not in df.columns:
            return df
        
        date_col = 'termination_date_datetime' if 'termination_date_datetime' in df.columns else 'termination_date'
        
        original_count = len(df)
        active_df = df[df[date_col].isna()]
        
        print(f"   Active merchants: {len(active_df):,} / {original_count:,}")
        return active_df
    
    def merge_datasets(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                      left_on: str, right_on: str, how: str = "inner") -> pd.DataFrame:
        """Merge two datasets with logging"""
        merged = pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=how)
        print(f"   Merged: {len(left_df):,} + {len(right_df):,} → {len(merged):,} rows")
        return merged
    
    def prepare_daily_sales(self) -> pd.DataFrame:
        """
        Prepare daily sales data for demand forecasting
        
        ENHANCED: Includes campaign data, taxonomy, and actual costs
        """
        print("\n" + "="*70)
        print("📊 PREPARING DAILY SALES DATA (Enhanced)")
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
        
        df["order_created_at"] = pd.to_datetime(df["order_created_at"], errors="coerce")
        df["date"] = df["order_created_at"].dt.date
        
        # Get actual costs from invoices
        invoice_items = self.load_invoice_items()
        
        if not invoice_items.empty and 'item_id' in invoice_items.columns:
            avg_costs = invoice_items.groupby('item_id').agg({
                'unit_price': 'mean'
            }).reset_index()
            avg_costs.columns = ['item_id', 'actual_unit_cost']
            
            df = df.merge(avg_costs, on='item_id', how='left')
            
            if 'cost' in df.columns:
                df['item_cost'] = df['actual_unit_cost'].fillna(df['cost'])
            else:
                df['item_cost'] = df['actual_unit_cost'].fillna(df['price'] * 0.6)
            
            print("   ✅ Merged actual costs from invoice data")
        else:
            if "cost" in df.columns:
                df["item_cost"] = df["cost"]
            else:
                df["item_cost"] = df["price"] * 0.6
                print("   ⚠️  Using estimated costs (60% of price)")
        
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
        
        # Add campaign flags
        campaigns = self.load_campaigns()
        
        daily_sales["is_campaign_active"] = 0
        daily_sales["campaign_type"] = None
        daily_sales["campaign_discount_pct"] = 0
        
        if not campaigns.empty:
            if "start_date_datetime" in campaigns.columns and "end_date_datetime" in campaigns.columns:
                campaigns["start_date"] = campaigns["start_date_datetime"].dt.date
                campaigns["end_date"] = campaigns["end_date_datetime"].dt.date
                
                for _, c in campaigns.iterrows():
                    mask = (
                        (daily_sales["date"] >= c["start_date"]) &
                        (daily_sales["date"] <= c["end_date"])
                    )
                    
                    daily_sales.loc[mask, "is_campaign_active"] = 1
                    
                    if 'type' in c and pd.notna(c['type']):
                        daily_sales.loc[mask, "campaign_type"] = c['type']
                    
                    if 'discount_value' in c and pd.notna(c['discount_value']):
                        daily_sales.loc[mask, "campaign_discount_pct"] = c['discount_value']
                
                print("   ✅ Added campaign activity flags")
        
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
        
        NEW METHOD: Calculate from Purchases - Sales
        Strategy Priority:
        1. Invoice Purchases - Order Sales = Current Stock (BEST)
        2. Inventory Reports (if available)
        3. Estimate from sales patterns (fallback)
        
        Returns:
            Inventory snapshot DataFrame
        """
        print("\n" + "="*70)
        print("📦 PREPARING INVENTORY SNAPSHOT (Purchase-Sales Method)")
        print("="*70)
        
        items = self.load_items()
        
        if items.empty:
            print("❌ Cannot prepare inventory - items table missing")
            return pd.DataFrame()
        
        # Keep only inventory-managed items
        inventory_items = items[items["manage_inventory"] == 1].copy()
        print(f"\n   Inventory-managed items: {len(inventory_items):,}")
        
        # ===== STRATEGY 1: Calculate from Purchases - Sales (NEW!) =====
        invoice_items = self.load_invoice_items()
        order_items = self.load_order_items()
        
        inventory_calculated = False
        
        if not invoice_items.empty and not order_items.empty:
            print("   ✅ Calculating inventory from purchases and sales")
            
            # STEP 1: Calculate total purchases per item
            if 'item_id' in invoice_items.columns and 'quantity' in invoice_items.columns:
                purchases = invoice_items.groupby('item_id').agg({
                    'quantity': 'sum',
                    'unit_price': 'mean'
                }).reset_index()
                purchases.columns = ['item_id', 'total_purchased', 'avg_purchase_cost']
                print(f"   📦 Purchases: {len(purchases):,} items, Total qty: {purchases['total_purchased'].sum():,.0f}")
            else:
                print("   ⚠️  Invoice items missing required columns")
                purchases = pd.DataFrame()
            
            # STEP 2: Calculate total sales per item
            if 'item_id' in order_items.columns and 'quantity' in order_items.columns:
                sales = order_items.groupby('item_id').agg({
                    'quantity': 'sum'
                }).reset_index()
                sales.columns = ['item_id', 'total_sold']
                print(f"   🛒 Sales: {len(sales):,} items, Total qty: {sales['total_sold'].sum():,.0f}")
            else:
                print("   ⚠️  Order items missing required columns")
                sales = pd.DataFrame()
            
            # STEP 3: Calculate current stock = Purchases - Sales
            if not purchases.empty and not sales.empty:
                # Start with inventory items
                inventory = inventory_items.copy()
                inventory = inventory.rename(columns={"id": "item_id"})
                
                # Merge purchases
                inventory = inventory.merge(purchases, on='item_id', how='left')
                
                # Merge sales
                inventory = inventory.merge(sales, on='item_id', how='left')
                
                # Calculate stock
                inventory['total_purchased'] = inventory['total_purchased'].fillna(0)
                inventory['total_sold'] = inventory['total_sold'].fillna(0)
                
                inventory['current_stock'] = (
                    inventory['total_purchased'] - inventory['total_sold']
                ).clip(lower=0).round(0).astype(int)
                
                # Use purchase cost as unit cost
                inventory['unit_cost'] = inventory['avg_purchase_cost'].fillna(30)
                
                inventory_calculated = True
                
                total_stock = inventory['current_stock'].sum()
                items_in_stock = (inventory['current_stock'] > 0).sum()
                out_of_stock = (inventory['current_stock'] == 0).sum()
                
                print(f"\n   📊 INVENTORY CALCULATION RESULTS:")
                print(f"      Formula: Purchases - Sales = Stock")
                print(f"      Total Stock: {total_stock:,.0f} units")
                print(f"      Items with Stock: {items_in_stock:,}")
                print(f"      Out of Stock: {out_of_stock:,}")
                print(f"      Stock Availability: {items_in_stock / len(inventory) * 100:.1f}%")
                
            elif not purchases.empty:
                # Only purchases available
                inventory = inventory_items.copy()
                inventory = inventory.rename(columns={"id": "item_id"})
                inventory = inventory.merge(purchases, on='item_id', how='left')
                
                inventory['current_stock'] = inventory['total_purchased'].fillna(0).round(0).astype(int)
                inventory['unit_cost'] = inventory['avg_purchase_cost'].fillna(30)
                inventory_calculated = True
                
                print(f"   ⚠️  Only purchase data available (no sales to subtract)")
                print(f"   Stock = Total Purchases: {inventory['current_stock'].sum():,.0f} units")
        
        # ===== STRATEGY 2: Use inventory reports if available =====
        if not inventory_calculated:
            inventory_reports = self.load_inventory_reports()
            
            if not inventory_reports.empty and 'item_id' in inventory_reports.columns:
                print("   📊 Using inventory reports as fallback")
                
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
                
                if 'quantity_on_hand' in inventory.columns:
                    inventory = inventory.rename(columns={"quantity_on_hand": "current_stock"})
                    inventory_calculated = True
        
        # ===== STRATEGY 3: Estimate from sales patterns (final fallback) =====
        if not inventory_calculated:
            print("   ⚠️  No purchase/invoice data - estimating from sales patterns")
            
            inventory = inventory_items.copy()
            inventory = inventory.rename(columns={"id": "item_id"})
            
            if self.sales_data is not None and len(self.sales_data) > 0:
                print("   Using sales velocity for estimation...")
                
                sales_stats = self.sales_data.groupby('item_id').agg({
                    'quantity_sold': ['mean', 'std', 'max', 'sum'],
                    'date': 'count'
                }).reset_index()
                
                sales_stats.columns = ['item_id', 'avg_daily_sales', 'std_daily_sales', 
                                      'max_daily_sales', 'total_sales', 'days_sold']
                
                inventory = inventory.merge(sales_stats, on='item_id', how='left')
                
                # Estimate: 7-14 days of stock
                np.random.seed(42)
                inventory['days_of_stock'] = np.random.uniform(7, 14, len(inventory))
                
                inventory['current_stock'] = (
                    inventory['avg_daily_sales'].fillna(5) * inventory['days_of_stock']
                ).round(0).astype(int)
                
                # Add variation
                random_factor = np.random.uniform(0.6, 1.4, len(inventory))
                inventory['current_stock'] = (
                    inventory['current_stock'] * random_factor
                ).round(0).astype(int).clip(lower=0)
                
                # Simulate stockouts (15%)
                out_of_stock_mask = np.random.random(len(inventory)) < 0.15
                inventory.loc[out_of_stock_mask, 'current_stock'] = 0
                
                print(f"   Estimated stock for {len(sales_stats)} items with sales history")
                
            else:
                print("   Using default random stock levels")
                inventory['current_stock'] = np.random.randint(0, 100, len(inventory))
        
        # ===== ADD COMMON FIELDS =====
        
        # Inventory variance tracking
        if "variance" in inventory.columns:
            inventory["stock_variance_rate"] = (
                inventory["variance"] / inventory["current_stock"]
            ).replace([np.inf, -np.inf], 0).fillna(0)
        else:
            inventory["stock_variance_rate"] = 0
        
        # Days in stock
        if 'days_in_stock' not in inventory.columns:
            inventory['days_in_stock'] = np.random.randint(0, 10, len(inventory))
        
        # Ensure unit_cost exists
        if 'unit_cost' not in inventory.columns:
            if 'price' in inventory.columns:
                inventory['unit_cost'] = inventory['price'] * 0.6
            else:
                inventory['unit_cost'] = 30
        
        # Ensure price exists
        if 'price' not in inventory.columns:
            if 'unit_cost' in inventory.columns:
                inventory['price'] = inventory['unit_cost'] / 0.6
            else:
                inventory['price'] = 50
        
        # Total value
        inventory['total_value'] = inventory['current_stock'] * inventory['unit_cost']
        
        # Reorder thresholds
        if 'reorder_point' not in inventory.columns:
            if 'avg_daily_sales' in inventory.columns:
                inventory['reorder_point'] = (inventory['avg_daily_sales'] * 3).fillna(10)
            else:
                inventory['reorder_point'] = 10
        
        if 'reorder_quantity' not in inventory.columns:
            if 'avg_daily_sales' in inventory.columns:
                inventory['reorder_quantity'] = (inventory['avg_daily_sales'] * 7).fillna(50)
            else:
                inventory['reorder_quantity'] = 50
        
        # ===== SUMMARY =====
        print(f"\n📊 INVENTORY SNAPSHOT SUMMARY:")
        print(f"   Total Items: {len(inventory):,}")
        print(f"   Items in Stock: {(inventory['current_stock'] > 0).sum():,}")
        print(f"   Out of Stock: {(inventory['current_stock'] == 0).sum():,}")
        print(f"   Average Stock: {inventory['current_stock'].mean():.1f} units")
        print(f"   Total Value: ${inventory['total_value'].sum():,.2f}")
        print(f"   Availability: {(inventory['current_stock'] > 0).sum() / len(inventory) * 100:.1f}%")
        print("="*70 + "\n")
        
        return inventory
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _estimate_shelf_life(self, row) -> int:
        """Estimate shelf life based on item characteristics"""
        item_type = str(row.get('type', '')).lower()
        title = str(row.get('title', '')).lower()
        
        # Highly perishable (3 days)
        if any(word in title for word in [
            'milk', 'cream', 'yogurt', 'cheese', 'salad', 
            'sandwich', 'fresh', 'juice', 'smoothie'
        ]):
            return 3
        
        # Perishable (7 days)
        if any(word in title for word in [
            'bread', 'pastry', 'cake', 'muffin', 'croissant',
            'coffee', 'latte', 'cappuccino'
        ]):
            return 7
        
        # Semi-perishable (30 days)
        if any(word in item_type for word in ['packaged', 'canned', 'bottled']):
            return 30
        
        # Long shelf life (90 days)
        if any(word in item_type for word in ['frozen', 'dry', 'grain', 'pasta', 'rice']):
            return 90
        
        # Non-perishable (60 days)
        if any(word in item_type for word in ['beverage', 'snack', 'condiment']):
            return 60
        
        return 14  # Default
    
    def validate_data_quality(self) -> Dict:
        """Run comprehensive data quality checks"""
        print("\n" + "="*70)
        print("🔍 DATA QUALITY VALIDATION")
        print("="*70 + "\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': []
        }
        
        # Check 1: Sales data
        if self.sales_data is not None:
            date_range = (self.sales_data['date'].max() - self.sales_data['date'].min()).days
            results['checks'].append({
                'check': 'Sales Data Coverage',
                'status': 'PASS' if date_range >= 90 else 'WARNING',
                'details': f'{date_range} days'
            })
        
        # Check 2: Invoice data
        invoice_items = self.load_invoice_items()
        results['checks'].append({
            'check': 'Invoice/Purchase Data',
            'status': 'PASS' if not invoice_items.empty else 'WARNING',
            'details': f'{len(invoice_items):,} records' if not invoice_items.empty else 'No data'
        })
        
        # Check 3: Order data
        order_items = self.load_order_items()
        results['checks'].append({
            'check': 'Order/Sales Data',
            'status': 'PASS' if not order_items.empty else 'FAIL',
            'details': f'{len(order_items):,} records' if not order_items.empty else 'No data'
        })
        
        # Check 4: Campaign data
        campaigns = self.load_campaigns()
        results['checks'].append({
            'check': 'Campaign Data',
            'status': 'PASS' if not campaigns.empty else 'INFO',
            'details': f'{len(campaigns):,} campaigns' if not campaigns.empty else 'No data'
        })
        
        # Check 5: BOM
        bom = self.load_bill_of_materials()
        results['checks'].append({
            'check': 'Bill of Materials',
            'status': 'PASS' if not bom.empty else 'WARNING',
            'details': f'{len(bom):,} recipes' if not bom.empty else 'No data'
        })
        
        # Print results
        for check in results['checks']:
            status_icon = {'PASS': '✅', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'FAIL': '❌'}
            icon = status_icon.get(check['status'], '•')
            print(f"{icon} {check['check']}: {check['details']}")
        
        print("\n" + "="*70 + "\n")
        
        return results
    
    def get_data_loading_summary(self) -> pd.DataFrame:
        """Get summary of all loaded tables"""
        if not self.loaded_tables:
            return pd.DataFrame()
        
        summary_data = []
        for filename, info in self.loaded_tables.items():
            summary_data.append({
                'filename': filename,
                'loaded': '✅' if info['loaded'] else '❌',
                'rows': info['rows'],
                'columns': info['columns'],
                'status': 'Success' if info['loaded'] else info.get('error', 'Failed')
            })
        
        return pd.DataFrame(summary_data).sort_values('loaded', ascending=False)

def create_item_mapping(self) -> Dict[int, str]:
    """
    Create mapping between simple IDs (1-40) and real product names
    
    Since there's no direct mapping, we need to create one intelligently
    """
    print("\n🗺️  CREATING ITEM NAME MAPPING...")
    
    # Load both item systems
    simple_items = self.load_items()  # IDs 1-40
    real_items = self.load_menu_items()  # Real product names
    
    if simple_items.empty or real_items.empty:
        print("   ⚠️  Cannot create mapping - missing item data")
        return {}
    
    # Create a mapping dictionary
    item_mapping = {}
    
    # Strategy 1: If same number of items, map sequentially
    if len(simple_items) <= len(real_items):
        print(f"   Mapping {len(simple_items)} simple items to {len(real_items)} real items")
        
        for i, (_, simple_row) in enumerate(simple_items.iterrows()):
            simple_id = simple_row['id']
            simple_title = simple_row['title']
            
            if i < len(real_items):
                real_title = real_items.iloc[i]['title']
                real_price = real_items.iloc[i].get('price', simple_row.get('price', 50))
                
                item_mapping[simple_id] = {
                    'simple_name': simple_title,
                    'real_name': real_title,
                    'real_price': real_price,
                    'mapping_type': 'sequential'
                }
                
                print(f"     {simple_title} (ID: {simple_id}) → {real_title}")
    
    # Store the mapping
    self.item_mapping = item_mapping
    
    print(f"   ✅ Created mapping for {len(item_mapping)} items")
    return item_mapping

def get_real_item_name(self, item_id: int) -> str:
    """Get real product name for a simple item ID"""
    if hasattr(self, 'item_mapping') and item_id in self.item_mapping:
        return self.item_mapping[item_id]['real_name']
    
    # Fallback to simple name
    if hasattr(self, 'sales_data') and 'title' in self.sales_data.columns:
        matches = self.sales_data[self.sales_data['item_id'] == item_id]['title'].unique()
        if len(matches) > 0:
            return matches[0]
    
    return f"Item {item_id}"
if __name__ == "__main__":

    # Example usage
    loader = EnhancedDataLoader("data/Inventory_Management")
    
    # Prepare sales data
    sales = loader.prepare_daily_sales()
    
    # Prepare inventory (now uses purchase-sales method!)
    inventory = loader.prepare_inventory_snapshot()
    
    # Validate data quality
    validation = loader.validate_data_quality()
    
    # Get loading summary
    summary = loader.get_data_loading_summary()
    print("\n📋 Data Loading Summary:")
    print(summary.to_string(index=False))