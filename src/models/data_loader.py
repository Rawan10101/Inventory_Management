"""
File: data_loader.py
Description: Handles loading and preprocessing of CSV data files for inventory/demand forecasting.
Dependencies: pandas, numpy
Author: Sample Team
"""

import pandas as pd
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
            df = pd.read_csv(file_path, parse_dates=parse_dates)
            print(f"Successfully loaded {filename}: {len(df)} rows")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_orders(self) -> pd.DataFrame:
        """Load orders, convert timestamp, and keep only closed orders."""
        orders = self.load_csv("fct_orders.csv")
        orders.columns = orders.columns.str.strip()
        # Convert 'created' timestamp to datetime
        orders['order_created_at'] = pd.to_datetime(orders['created'], unit='s')
        # Keep only closed orders
        orders = orders[orders['status'] == 'Closed']
        return orders

    def load_order_items(self) -> pd.DataFrame:
        return self.load_csv("fct_order_items.csv")

    def load_inventory_reports(self) -> pd.DataFrame:
        return self.load_csv("fct_inventory_reports.csv", parse_dates=["report_date"])

    def load_items(self) -> pd.DataFrame:
        return self.load_csv("dim_items.csv")

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
        # orders = orders.rename(columns={"created": "order_created_at"})

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

        print("Prepared daily sales dataset (ML-ready)")
        return daily_sales

    def prepare_inventory_snapshot(self) -> pd.DataFrame:
        """
        Prepares current inventory snapshot for items managed in inventory.
        """
        inventory = self.load_inventory_reports()
        items = self.load_items()

        # Keep only inventory-managed items
        inventory_items = items[items["manage_inventory"] == 1]

        inventory = inventory.merge(
            inventory_items[["id", "title"]],
            left_on="item_id",
            right_on="id",
            how="inner"
        )

        inventory = inventory.rename(columns={"quantity_on_hand": "current_stock"})

        print("Prepared inventory snapshot (ML-ready)")
        return inventory
