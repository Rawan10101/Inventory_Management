"""
File: data_loader.py
Description: Handles loading and preprocessing of CSV data files.
Dependencies: pandas, numpy
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""

import pandas as pd
from typing import Optional, List


class DataLoader:
    """
    Handles loading and basic preprocessing of CSV data files.
    
    Attributes:
        data_path (str): Path to the data directory.
        data (pd.DataFrame): The loaded dataset.
    
    Methods:
        load_csv(filename): Loads a CSV file into a DataFrame.
        merge_datasets(datasets): Merges multiple datasets.
        filter_active_merchants(df): Filters for active merchants only.
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
        
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = f"{self.data_path}/{filename}"
        
        try:
            df = pd.read_csv(file_path, parse_dates=parse_dates)
            print(f"Successfully loaded {filename}: {len(df)} rows")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def merge_datasets(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                      on: str, how: str = 'inner') -> pd.DataFrame:
        """
        Merges two datasets on a common key.
        
        Args:
            left_df (pd.DataFrame): Left dataset.
            right_df (pd.DataFrame): Right dataset.
            on (str): Column name to merge on.
            how (str): Type of merge ('inner', 'left', 'right', 'outer').
        
        Returns:
            pd.DataFrame: Merged dataset.
        """
        merged_df = pd.merge(left_df, right_df, on=on, how=how)
        print(f"Merged datasets: {len(merged_df)} rows")
        return merged_df
    
    def filter_active_merchants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters DataFrame to include only active merchants.
        
        Args:
            df (pd.DataFrame): DataFrame containing merchant data with 'termination_date' column.
        
        Returns:
            pd.DataFrame: Filtered DataFrame with active merchants only.
        """
        # Active merchants have NULL termination_date
        active_df = df[df['termination_date'].isnull()]
        print(f"Active merchants: {len(active_df)} out of {len(df)}")
        return active_df
