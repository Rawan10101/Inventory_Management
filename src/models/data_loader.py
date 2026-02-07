"""
File: data_loader.py
Description: Handles loading and preprocessing of CSV data files for inventory/demand forecasting.
Dependencies: pandas, numpy
Author: Sample Team
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class DataLoader:
    """
    Handles loading and basic preprocessing of CSV data files for inventory/demand forecasting.
    
    Attributes:
        data_path (str): Path to the data directory.
        data (pd.DataFrame): The loaded dataset.
    """

    def __init__(self, data_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the data directory.
        """
        self.data_path = data_path
        self.data = None
        self.sales_data = None

    def load_csv(self, filename: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.
        
        Args:
            filename (str): Name of the CSV file to load.
            parse_dates (List[str], optional): Column names to parse as dates.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        file_path = f"{self.data_path}/{filename}"
        try:
            df = self._read_csv_with_fallback(file_path, parse_dates=parse_dates)
            print(f"Successfully loaded {filename}: {len(df)} rows")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

    def _read_csv_with_fallback(self, file_path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        for encoding in encodings:
            try:
                return pd.read_csv(
                    file_path,
                    parse_dates=parse_dates,
                    low_memory=False,
                    encoding=encoding,
                )
            except UnicodeDecodeError:
                continue
        return pd.read_csv(
            file_path,
            parse_dates=parse_dates,
            low_memory=False,
            encoding="latin1",
            encoding_errors="ignore",
        )

    def load_orders(self) -> pd.DataFrame:
        """Load orders, convert timestamp, and keep only closed orders."""
        orders = self.load_csv("fct_orders.csv")
        orders.columns = orders.columns.str.strip()
        # Convert 'created' timestamp to datetime
        orders['order_created_at'] = pd.to_datetime(orders['created'], unit='s', errors='coerce')
        # Keep only closed orders
        orders = orders[orders['status'] == 'Closed']
        return orders

    def load_order_items(self) -> pd.DataFrame:
        return self.load_csv("fct_order_items.csv")

    def load_inventory_reports(self) -> pd.DataFrame:
        """Load inventory reports if available."""
        try:
            df = self.load_csv("fct_inventory_reports.csv")
            # Try to parse date columns
            if 'report_date' in df.columns:
                df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("Warning: fct_inventory_reports.csv not found or empty")
            return pd.DataFrame()

    def load_items(self) -> pd.DataFrame:
        """Load dim_items - raw inventory items catalog."""
        return self.load_csv("dim_items.csv")

    def load_menu_items(self) -> pd.DataFrame:
        """Load dim_menu_items - the merchant's product setup."""
        try:
            return self.load_csv("dim_menu_items.csv")
        except FileNotFoundError:
            print("Warning: dim_menu_items.csv not found")
            return pd.DataFrame()

    def load_bill_of_materials(self) -> pd.DataFrame:
        """Load dim_bill_of_materials - recipe ingredient breakdown."""
        try:
            return self.load_csv("dim_bill_of_materials.csv")
        except FileNotFoundError:
            print("Warning: dim_bill_of_materials.csv not found")
            return pd.DataFrame()

    def load_places(self) -> pd.DataFrame:
        return self.load_csv("dim_places.csv", parse_dates=["contract_start", "termination_date"])

    def filter_active_merchants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only merchants with no termination date."""
        active_df = df[df["termination_date"].isna()]
        print(f"Active merchants: {len(active_df)}")
        return active_df

    def merge_datasets(self, left_df, right_df, left_on, right_on, how="inner"):
        merged = pd.merge(
            left_df,
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=how
        )
        print(f"Merged datasets: {len(merged)} rows")
        return merged

    def prepare_daily_sales(self) -> pd.DataFrame:
        """
        Creates daily item level sales for demand forecasting,
        including item and place metadata.
        """
        orders = self.load_orders()
        order_items = self.load_order_items()

        # Merge order-level info into item-level
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

        # Aggregate quantity and revenue per day/item/place
        daily_sales = (
            df.groupby(["place_id", "item_id", "date"])
              .agg(quantity_sold=("quantity", "sum"),
                   revenue=("price", "sum"))
              .reset_index()
        )

        # Merge item metadata
        items = self.load_items()
        daily_sales = daily_sales.merge(
            items[["id", "title", "type", "manage_inventory"]],
            left_on="item_id",
            right_on="id",
            how="left"
        )

        # Merge place metadata
        places = self.load_places()
        active_places = self.filter_active_merchants(places)
        daily_sales = daily_sales.merge(
            active_places[["id", "title"]],
            left_on="place_id",
            right_on="id",
            how="left",
            suffixes=("", "_place")
        )

        # Store for later use in inventory snapshot
        self.sales_data = daily_sales

        print("Prepared daily sales dataset (ML-ready)")
        return daily_sales

    def prepare_inventory_snapshot(self, allow_mock: bool = False) -> pd.DataFrame:
        """
        Prepares current inventory snapshot for items managed in inventory.
        Creates realistic mock inventory based on sales patterns if actual data unavailable.
        """
        inventory_reports = self.load_inventory_reports()
        items = self.load_items()

        # Keep only inventory-managed items (fallback to all items if flag is sparse)
        if "manage_inventory" in items.columns:
            inventory_items = items[items["manage_inventory"] == 1].copy()
            if inventory_items.empty or len(inventory_items) < max(5, int(len(items) * 0.01)):
                inventory_items = items.copy()
        else:
            inventory_items = items.copy()

        if not inventory_reports.empty and 'item_id' in inventory_reports.columns and len(inventory_reports) > 0:
            # Use actual inventory reports if available
            inventory = inventory_reports.merge(
                inventory_items[["id", "title"]],
                left_on="item_id",
                right_on="id",
                how="inner"
            )
            inventory = inventory.rename(columns={"quantity_on_hand": "current_stock"})
            print("✅ Prepared inventory snapshot from fct_inventory_reports (ML-ready)")
        else:
            if not allow_mock:
                print("⚠️  fct_inventory_reports missing or invalid - inventory snapshot not generated")
                return pd.DataFrame()

            print("⚠️  fct_inventory_reports missing - mock inventory is disabled by policy")
            return pd.DataFrame()

        print(f"📦 Inventory snapshot ready: {len(inventory)} items")
        return inventory

    def prepare_item_consumption(self) -> pd.DataFrame:
        """
        Prepares item consumption data by linking orders to raw materials
        through bill of materials (recipes).
        """
        try:
            order_items = self.load_order_items()
            bom = self.load_bill_of_materials()
            
            if bom.empty:
                print("Warning: Bill of materials not available")
                return pd.DataFrame()
            
            # Join order items with BOM to get raw material consumption
            consumption = order_items.merge(
                bom,
                left_on="item_id",
                right_on="menu_item_id",
                how="inner"
            )
            
            # Calculate raw material consumption
            consumption['raw_material_consumed'] = (
                consumption['quantity'] * consumption['ingredient_quantity']
            )
            
            print("Prepared item consumption data (ML-ready)")
            return consumption
            
        except Exception as e:
            print(f"Error preparing consumption data: {e}")
            return pd.DataFrame()
